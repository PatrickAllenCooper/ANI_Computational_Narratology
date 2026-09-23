"""
scripts/make_iclr_figures.py -- figure and table sources for papers/iclr2027, generated from
the committed artefacts only (every number carries a % source comment in the output).

  figures/fig_judge_panel.tex    pillar 1: NoT minus CoT validation drop per generator under
                                 every judge, two panels. Left, the 17.10 same-day regeneration
                                 of both prompts at 4,096 tokens (headline_rep4k_readout.json,
                                 the registered headline); right, the original responses of the
                                 17.8 panel (judge_panel_full_oeq_<judge>.json)
  figures/tab_routing.tex        pillar 3: collective, routed, routing gain and the sensor's
                                 precision and recall per condition (verify_pillar3_headline_*.json),
                                 the same model's solo, and the paired collective minus solo and
                                 routed collective minus solo (collective_vs_solo_paired.json)
  figures/tab_form_main.tex      main-text form / content / persona / length table, prompts as rows
                                 and the five judges as columns (narrative_form_readout.json)
  figures/tab_form.tex           appendix form table with readings and Holm daggers
                                 (narrative_form_readout.json, multiplicity_mde.json)
  figures/tab_rep4k.tex          17.10 replicate by judge with Holm daggers
                                 (headline_rep4k_readout.json, multiplicity_mde.json)
  figures/tab_pushback.tex       17.9 capitulation (pushback_readout_*.json)
  figures/tab_seat.tex           16.28 narrated minus plain seats (seat_scaffold_comparison.json)
  figures/tab_runs.tex           run-to-run drops from the earlier scratch readout
                                 (run_stability_readout.json; kept while the appendix inputs it)
  figures/tab_run_stability.tex  run-to-run drops from the promoted artefact (run_stability.json)
  figures/tab_z3.tex             asker shielding, signal detection (z3_sdt_<model>.json)
  figures/tab_interjudge.tex     inter-judge kappa against the production judge
  figures/tab_knockout_readouts.tex  17.3 knockout descriptives and Manski brackets (readouts_17_3_17_5.json)
                                 (interjudge_kappa.json) with its test-retest row
                                 (judge_test_retest.json)

Missing artefacts are skipped and listed in the output header, never silently zero-filled.

Run:  python -m scripts.make_iclr_figures
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("divergence_study_outputs")
FIG = Path("papers/iclr2027/figures")
GENS = [("claude-haiku-4-5", "haiku"), ("claude-sonnet-4-6", "sonnet"), ("gpt-5.4-nano", "nano"),
        ("grok-4-1-fast-reasoning", "grok"), ("Llama-3.3-70B-Instruct", "Llama"),
        ("Mistral-Large-3-2", "Mistral"), ("DeepSeek-V4-Pro", "DeepSeek")]
# (key, legend label, colour, pgfplots mark, mark fill). The production judge is the filled black
# circle and the other four are open shapes, so the judge panel reads in greyscale print (violet and
# red, and teal and grey, collapse to one tone when filled).
JUDGES = [("claude-haiku-4-5", "haiku (production)", "black", "*", "black"),
          ("gpt-4o", "gpt-4o", "orange!85!black", "square", "white"),
          ("gpt-5.4-nano", "nano", "violet!80!black", "triangle", "white"),
          ("Llama-3.3-70B-Instruct", "Llama", "gray!60!black", "diamond", "white"),
          ("grok-4-1-fast-reasoning", "grok", "red!70!black", "pentagon", "white")]
CONDITIONS = [("grok", "grok, edges on (primary)"), ("noedge", "grok, edges off"),
              ("haiku_249", "haiku"), ("llama", "Llama"), ("deepseek", "DeepSeek"), ("mistral", "Mistral")]
# verify_pillar3_headline condition -> collective_vs_solo_paired.json condition (None: no solo pairing;
# the edges-off community has no solo of its own in that artefact).
SOLO_COND = {"grok": "grok", "noedge": None, "haiku_249": "haiku", "llama": "llama",
             "deepseek": "deepseek", "mistral": "mistral"}
DAGGER = "$^{\\dagger}$"


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _judge_series_original():
    """Right panel: the original responses, judge_panel_full_oeq_<judge>.json (prereg 17.8 RESULTS)."""
    series, missing = [], []
    for jk, jl, *_ in JUDGES:
        d = _load(OUT / f"judge_panel_full_oeq_{jk}.json")
        pts = []
        if d is not None:
            pg = d["judges"][jk]["per_generator"]
            for gi, (gk, gl) in enumerate(GENS):
                c = pg.get(gk, {}).get("narrative_cot")
                if not c or c.get("drop") is None:
                    continue
                pts.append((gi, 100 * c["drop"], 100 * c["lo"], 100 * c["hi"], gl))
        if not pts:
            missing.append(f"judge_panel_full_oeq_{jk}.json")
            continue
        series.append((jk, jl, pts))
    return series, missing


def _judge_series_rep4k():
    """Left panel: both prompts regenerated in one run at a 4,096-token cap, headline_rep4k_readout.json
    per_generator.<gen>.per_judge.<judge> (prereg Addendum 17.10 RESULTS), the registered headline."""
    d = _load(OUT / "headline_rep4k_readout.json")
    if d is None:
        return [], ["headline_rep4k_readout.json"]
    series, missing = [], []
    for jk, jl, *_ in JUDGES:
        pts = []
        for gi, (gk, gl) in enumerate(GENS):
            c = d["per_generator"].get(gk, {}).get("per_judge", {}).get(jk)
            if not c or c.get("drop") is None:
                continue
            pts.append((gi, 100 * c["drop"], 100 * c["lo"], 100 * c["hi"], gl))
        if not pts:
            missing.append(f"headline_rep4k_readout.json per_judge.{jk}")
            continue
        series.append((jk, jl, pts))
    return series, missing


def _judge_axis(series, name, title, first):
    """One axis of the two-panel judge figure. The first axis carries the row labels and the legend,
    which sits in the upper left with no fill: the lower left holds Llama's and Mistral's drops in both
    panels, and an opaque box in the upper left would cover the nano row's production-judge marker.
    Marks are 2.3pt (2.5pt touches neighbouring rows at this height) with a per-judge row offset."""
    style = {jk: (col, mark, fill) for jk, _jl, col, mark, fill in JUDGES}
    k = len(series)
    opts = [f"name={name}", "scale only axis", f"title={{{title}}}",
            "title style={font=\\small, yshift=-3pt}", "width=0.40\\columnwidth", "height=5.0cm",
            "xmin=-85", "xmax=25", "xlabel={NoT $-$ CoT, validation rate (points)}",
            "xlabel style={font=\\small}", f"ytick={{{','.join(str(i) for i in range(len(GENS)))}}}",
            "ymin=-0.6", "ymax=6.6", "y dir=reverse", "xmajorgrids", "axis lines*=left",
            "tick label style={font=\\small}"]
    if first:
        opts += [f"yticklabels={{{','.join(g[1] for g in GENS)}}}",
                 "legend style={font=\\scriptsize, at={(0.02,0.99)}, anchor=north west, draw=none, "
                 "fill=none, row sep=-3pt}", "legend cell align=left"]
    else:
        opts += ["yticklabels={,,}", "at={(left.east)}", "anchor=west", "xshift=0.45cm"]
    lines = ["\\begin{axis}[" + ", ".join(opts) + "]"]
    for gi in range(0, len(GENS), 2):  # light bands on alternate generator rows
        lines.append(f"\\fill[gray!9] (axis cs:-85,{gi - 0.5}) rectangle (axis cs:25,{gi + 0.5});")
    lines.append("\\addplot[black, dashed, forget plot] coordinates {(0,-0.6) (0,6.6)};")
    for si, (jk, jl, pts) in enumerate(series):
        off = (si - (k - 1) / 2) * 0.16
        col, mark, fill = style[jk]
        lines.append(f"\\addplot[only marks, mark={mark}, mark size=2.3pt, color={col}, "
                     f"mark options={{fill={fill}, line width=0.7pt}}, error bars/.cd, x dir=both, "
                     "x explicit, error bar style={line width=0.55pt}]")
        lines.append("  coordinates {")
        for gi, dr, lo, hi, gl in pts:
            lines.append(f"    ({dr:.1f},{gi + off:.2f}) -= ({dr - lo:.1f},0) += ({hi - dr:.1f},0)"
                         f"  % {gl}: {dr:+.1f} [{lo:+.1f}, {hi:+.1f}]")
        lines.append("  };")
        if first:
            lines.append(f"\\addlegendentry{{{jl}}}")
    lines.append("\\end{axis}")
    return lines


def judge_panel_figure() -> str:
    rep, m1 = _judge_series_rep4k()
    orig, m2 = _judge_series_original()
    lines = ["% AUTO-GENERATED by scripts/make_iclr_figures.py. Left: headline_rep4k_readout.json (prereg",
             "% Addendum 17.10 RESULTS, both prompts regenerated in one run at a 4,096-token cap, the",
             "% registered headline). Right: judge_panel_full_oeq_<judge>.json (17.8 RESULTS, the original",
             "% responses at their caps). Production judge = filled black circle; other judges open shapes.",
             f"% missing at generation time: {(m1 + m2) or 'none'}",
             "\\begin{tikzpicture}"]
    lines += _judge_axis(rep, "left", "Regenerated, 4,096-token cap", True)
    lines += _judge_axis(orig, "right", "Original responses", False)
    lines += ["\\end{tikzpicture}"]
    return "\n".join(lines) + "\n"


def _pts(c):
    """collective_vs_solo_paired.json paired statistic (points, points_lo, points_hi) in math mode."""
    return f"${c['points']:+.1f}$ $[{c['points_lo']:+.1f}, {c['points_hi']:+.1f}]$"


def routing_table() -> str:
    """Main-text tab-routing. Per condition: debates, fire rate, the same model's solo (standard
    majority-of-3 on the paired framing x item cells), collective and routed accuracy, routed minus
    collective (verify_pillar3_headline_<cond>.json recomputed), the paired collective minus solo and
    routed collective minus solo (collective_vs_solo_paired.json, debate-weighted, sonnet judge on
    flagged debates), and the sensor's precision and recall on the one-loser stratum."""
    solo = _load(OUT / "collective_vs_solo_paired.json")
    rows, missing = [], []
    for ck, cl in CONDITIONS:
        v = _load(OUT / f"verify_pillar3_headline_{ck}.json")
        if v is None:
            missing.append(ck)
            continue
        r = v["recomputed"]
        # Sensor operating point on the one-loser stratum (verdicts blaming exactly one party):
        # precision = share of flagged verdicts that are wrong, recall = share of wrong verdicts flagged.
        wf = r["ol_fired"] * r["ol_wrong_fired"]
        wu = (r["ol_n"] - r["ol_fired"]) * r["ol_wrong_unfired"]
        prec = r["ol_wrong_fired"]
        rec = wf / (wf + wu)
        sk = SOLO_COND.get(ck)
        p = solo["conditions"][sk]["paired"] if (solo is not None and sk and sk in solo["conditions"]) else None
        if sk and p is None:
            missing.append(f"collective_vs_solo_paired.json conditions.{sk}")
        rows.append((cl, r["collective"], r["routed"], r["delta"], r["lo"], r["hi"], r["fire"], r["n"],
                     v.get("all_agree"), prec, rec, r["ol_n"], r["ol_fired"], wf, wu, sk, p))
    out = ["% AUTO-GENERATED by scripts/make_iclr_figures.py from verify_pillar3_headline_<cond>.json",
           "% (independent recomputation; sonnet standard majority-of-3 judge on flagged debates) and",
           "% collective_vs_solo_paired.json (scripts/analyze_collective_vs_solo.py, post hoc, zero spend,",
           "% prereg block 'Zero-spend post hoc analyses for review round 1').",
           "% solo = conditions.<c>.paired.solo_acc_paired_cells (the model's standard majority-of-3 on the",
           "% paired framing x item cells, = routing_certification acc_base); coll. - solo and routed - solo =",
           "% conditions.<c>.paired.{collective,routed}_minus_solo_debate_weighted.points [points_lo, points_hi],",
           "% paired per debate, 4,000-draw item-clustered bootstrap, seed 101. 'routed' there is the ROUTED",
           "% COLLECTIVE (S2 unless the counter fires, then the sonnet judge) against the solo, NOT the 16.22 R1",
           "% system that routes the solo's own verdicts by the collective's flags (routing_certification*.json:",
           "% grok +2.4 [+0.5, +4.5], haiku +3.1 [+1.2, +5.1]). Edges off has no solo pairing (blank).",
           "% precision = ol_wrong_fired; recall = ol_fired*ol_wrong_fired / (ol_fired*ol_wrong_fired + (ol_n-ol_fired)*ol_wrong_unfired),",
           "% both on verdicts that blame exactly one party (the one-loser stratum, prereg 16.15.1 Rule 1).",
           f"% missing at generation time: {missing or 'none'}",
           "% The file carries its own resizebox and 3pt column padding: ten columns run to about 500pt at",
           "% \\small against the 397pt text width. Remove the wrapper if the table is restructured.",
           "{\\setlength{\\tabcolsep}{3pt}\\resizebox{\\textwidth}{!}{%",
           "\\begin{tabular}{lrrrrrrrrr}", "\\toprule",
           "condition & $n$ & fire & solo & collective & routed & routed $-$ collective & "
           "collective $-$ solo & routed $-$ solo & prec. / recall \\\\", "\\midrule"]
    for cl, coll, rt, d, lo, hi, fire, n, agree, prec, rec, ol_n, ol_fired, wf, wu, sk, p in rows:
        if p is not None:
            solo_c = f"{p['solo_acc_paired_cells']:.3f}"
            cms = _pts(p["collective_minus_solo_debate_weighted"])
            rms = _pts(p["routed_minus_solo_debate_weighted"])
            note = (f"; solo cells {p['n_paired_cells']}, paired debates {p['n_paired_debates']}, "
                    f"items {p['n_items']}, debates without solo {p['n_debates_without_solo']}")
        else:
            solo_c, cms, rms, note = "", "", "", "; no solo pairing in collective_vs_solo_paired.json"
        out.append(f"{cl} & {n} & {fire:.2f} & {solo_c} & {coll:.3f} & {rt:.3f} & "
                   f"${100 * d:+.1f}$ $[{100 * lo:+.1f}, {100 * hi:+.1f}]$ & {cms} & {rms} & "
                   f"{prec:.2f} / {rec:.2f} \\\\"
                   f"  % verify all_agree={agree}; one-party {ol_n}, flagged {ol_fired}, "
                   f"wrong flagged {round(wf)}, wrong unflagged {round(wu)}{note}")
    out += ["\\bottomrule", "\\end{tabular}}}"]
    return "\n".join(out) + "\n"


JUDGE_SHORT = {"claude-haiku-4-5": "haiku (prod.)", "gpt-4o": "gpt-4o", "gpt-5.4-nano": "nano",
               "Llama-3.3-70B-Instruct": "Llama", "grok-4-1-fast-reasoning": "grok"}


def _ci(c, scale=100):
    # Interval in math mode so its minus signs match the point estimate's (appendix tables).
    return f"${scale * c['point']:+.1f}$ $[{scale * c['lo']:+.1f}, {scale * c['hi']:+.1f}]$"


def _holm_cells(table_key):
    """multiplicity_mde.json tables.<table_key>.per_judge.<judge>.cells.<generator>, or None if absent."""
    d = _load(OUT / "multiplicity_mde.json")
    if d is None or table_key not in d.get("tables", {}):
        return None
    return {j: pj["cells"] for j, pj in d["tables"][table_key]["per_judge"].items()}


def _fails_holm(c) -> bool:
    """A cell whose raw 95 percent interval excludes zero but which the exact within-judge Holm
    step-down does not keep (the only cells where the correction changes anything)."""
    return bool(c and c.get("ci_excludes_0") and not c.get("survives_holm_exact", True))


# tab_form rows -> the per-generator family in multiplicity_mde.json tables.table6_form (m = 4 generators)
FORM_FAMILY = {"NoT": "not_rep_drop", "checklist": "checklist_drop",
               "narrative_only": "narrative_only_drop", "persona": "persona_drop"}


def form_table() -> str:
    d = _load(OUT / "narrative_form_readout.json")
    if d is None:
        return "% narrative_form_readout.json not present\n"
    holm = _load(OUT / "multiplicity_mde.json")
    fam = holm["tables"]["table6_form"]["per_judge"] if holm else None
    out = ["% AUTO-GENERATED from narrative_form_readout.json (17.7, 17.7 amendment 2, 17.4; same-day",
           "% replicates standard_cot_rep / narrative_cot_rep as comparators; pooled over haiku, grok, nano, Llama;",
           "% item-clustered bootstrap). Drop = arm minus CoT, points; share = arm's drop over NoT's drop.",
           "% Dagger: multiplicity_mde.json tables.table6_form.per_judge.<judge>.<family>.cells, families",
           "% not_rep_drop / checklist_drop / narrative_only_drop / persona_drop over the four generators within",
           "% the judge (m = 4, exact p). The rows here are POOLED and the artefact tests each generator, so the",
           "% dagger marks a pooled cell whose family has at least one generator with a raw interval excluding",
           "% zero that the Holm step-down does not keep; the per-row comment gives survivors of m and the",
           "% generators removed. A missing artefact leaves the column undaggered.",
           f"% multiplicity_mde.json present: {holm is not None}",
           "\\begin{tabular}{llrrl}", "\\toprule",
           "judge & arm & drop from CoT & share of NoT & reading \\\\", "\\midrule"]
    any_dagger = False

    def mark(judge, family):
        nonlocal any_dagger
        if not fam or judge not in fam or family not in fam[judge]:
            return "", ""
        f = fam[judge][family]
        removed = [gl for gk, gl in GENS if _fails_holm(f["cells"].get(gk))]
        note = (f"  % Holm {family}: {f['n_survive_holm']} of {f['m']} survive, raw excluding zero "
                f"{f['n_raw_ci_excludes_0']}" + (f", removed {removed}" if removed else ""))
        if removed:
            any_dagger = True
            return DAGGER, note
        return "", note
    for j, jr in d["judges"].items():
        f = jr["form_pooled"]
        ck, no, pc = f["checklist"], f["narrative_only"], jr.get("persona_pooled", {})
        if ck.get("drop_arm") is None:
            continue
        dg, nt = mark(j, FORM_FAMILY["NoT"])
        out.append(f"{JUDGE_SHORT.get(j, j)} & NoT & {_ci(ck['drop_not'])}{dg} & 1.00 & \\\\{nt}")
        dg, nt = mark(j, FORM_FAMILY["checklist"])
        out.append(f" & checklist, same content & {_ci(ck['drop_arm'])}{dg} & {ck['share']['point']:.2f} & {jr['form_reading']['checklist']} \\\\{nt}")
        dg, nt = mark(j, FORM_FAMILY["narrative_only"])
        out.append(f" & narrative only & {_ci(no['drop_arm'])}{dg} & {no['share']['point']:.2f} & {jr['form_reading']['narrative_only']} \\\\{nt}")
        if pc.get("drop_arm") is not None:
            dg, nt = mark(j, FORM_FAMILY["persona"])
            out.append(f" & persona only & {_ci(pc['drop_arm'])}{dg} & {pc['share']['point']:.2f} & {jr.get('persona_reading', '')} \\\\{nt}")
        out.append("\\addlinespace")
    out += ["\\bottomrule"]
    if any_dagger:
        out.append("\\multicolumn{5}{l}{\\scriptsize $^{\\dagger}$ a per-generator interval behind this pooled cell "
                   "fails the Holm step-down within the judge.} \\\\")
    out += ["\\end{tabular}"]
    return "\n".join(out) + "\n"


def rep4k_table() -> str:
    d = _load(OUT / "headline_rep4k_readout.json")
    if d is None:
        return "% headline_rep4k_readout.json not present\n"
    holm = _holm_cells("table5_rep4k")
    # Columns in the order of the five-judge panel (JUDGES), not the artefact's order, so this table
    # reads against tab-app-judge-drops and Figure 1; every cell carries its item-clustered interval.
    order = [jk for jk, *_ in JUDGES]
    judges = [j for j in order if j in d["judges"]] + [j for j in d["judges"] if j not in order]
    out = ["% AUTO-GENERATED from headline_rep4k_readout.json (17.10: same-day, both arms at 4,096 tokens).",
           "% Dagger: multiplicity_mde.json tables.table5_rep4k.per_judge.<judge>.cells.<gen> with ci_excludes_0",
           "% true and survives_holm_exact false (Holm step-down over the seven generators within the judge,",
           "% exact item-bootstrap p). A missing artefact leaves every cell undaggered.",
           f"% multiplicity_mde.json present: {holm is not None}",
           "\\begin{tabular}{l" + "r" * len(judges) + "l}", "\\toprule",
           "generator & " + " & ".join(JUDGE_SHORT.get(j, j) for j in judges) + " & reading \\\\", "\\midrule"]
    daggered = []
    for gk, gl in GENS:
        v = d["per_generator"].get(gk)
        if not v:
            continue
        cells = []
        for j in judges:
            c = v["per_judge"].get(j, {})
            if c.get("drop") is None:
                cells.append("--")
                continue
            dg = ""
            if holm and _fails_holm(holm.get(j, {}).get(gk)):
                dg = DAGGER
                daggered.append(f"{JUDGE_SHORT.get(j, j)} judge x {gl} (adj. p {holm[j][gk]['holm_adj_p_exact']:.3f})")
            cells.append(f"${100 * c['drop']:+.1f}$ $[{100 * c['lo']:+.1f}, {100 * c['hi']:+.1f}]${dg}")
        out.append(f"{gl} & " + " & ".join(cells) + f" & {v['reading'].replace('HEADLINE-', '')} \\\\")
    out += ["\\bottomrule"]
    if daggered:
        out.append(f"\\multicolumn{{{len(judges) + 2}}}{{l}}{{\\footnotesize $^{{\\dagger}}$ interval excludes zero but "
                   "a Holm step-down over the seven generators within the judge does not keep it.} \\\\"
                   f"  % daggered: {'; '.join(daggered)}")
    out += ["\\end{tabular}"]
    return "\n".join(out) + "\n"


def form_main_table() -> str:
    """Main-text tab-form-main: prompts as rows, the five judges as columns, every cell from
    narrative_form_readout.json. Block 1 is prompt minus same-day CoT pooled over haiku, grok, nano
    and Llama (form_pooled / persona_pooled drop_arm, drop_not). Block 2 is NoT minus length-matched
    CoT per generator (length_per_generator gap); the length prompt ran on haiku and grok only."""
    d = _load(OUT / "narrative_form_readout.json")
    if d is None:
        return "% narrative_form_readout.json not present\n"
    order = [jk for jk, *_ in JUDGES]
    judges = [j for j in order if j in d["judges"]] + [j for j in d["judges"] if j not in order]
    out = ["% AUTO-GENERATED from narrative_form_readout.json (17.7, 17.7 amendment 2, 17.4). Block 1: prompt",
           "% minus same-day CoT (standard_cot_rep), points, pooled over haiku, grok, nano and Llama, item-clustered",
           "% bootstrap (form_pooled.<arm>.drop_arm, form_pooled.checklist.drop_not, persona_pooled.drop_arm).",
           "% Block 2: NoT minus length-matched CoT per generator (length_per_generator.<gen>.gap); the length",
           "% prompt ran on haiku and grok only, so it has no pooled value. Columns in the 17.8 panel order.",
           "\\begin{tabular}{l" + "r" * len(judges) + "}", "\\toprule",
           "prompt $-$ same-day CoT & " + " & ".join(JUDGE_SHORT.get(j, j) for j in judges) + " \\\\", "\\midrule",
           ]

    def row(label, pick):
        cells = []
        for j in judges:
            c = pick(d["judges"][j])
            cells.append("--" if c is None else _ci(c))
        out.append(f"{label} & " + " & ".join(cells) + " \\\\")

    row("NoT", lambda jr: jr["form_pooled"]["checklist"].get("drop_not"))
    row("checklist, same content", lambda jr: jr["form_pooled"]["checklist"].get("drop_arm"))
    row("narrative only", lambda jr: jr["form_pooled"]["narrative_only"].get("drop_arm"))
    row("persona only", lambda jr: jr.get("persona_pooled", {}).get("drop_arm"))
    out.append("\\midrule")
    for gk, gl in (("claude-haiku-4-5", "NoT $-$ length-matched CoT, haiku"),
                   ("grok-4-1-fast-reasoning", "NoT $-$ length-matched CoT, grok")):
        row(gl, lambda jr, gk=gk: jr.get("length_per_generator", {}).get(gk, {}).get("gap"))
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def pushback_table() -> str:
    rows = []
    for m, lab in (("gpt-5.4-nano", "nano"), ("Llama-3.3-70B-Instruct", "Llama"), ("grok-4-1-fast-reasoning", "grok")):
        short = {"gpt-5.4-nano": "nano", "Llama-3.3-70B-Instruct": "llama", "grok-4-1-fast-reasoning": "grok"}[m]
        d = _load(OUT / f"pushback_readout_{short}.json")
        if d is None:
            continue
        pm = d["per_model"][m]
        rows.append((lab, pm["n_common_support"], pm.get("cap"), pm.get("inst"), pm.get("net")))
    pooled = _load(OUT / "pushback_readout_pooled.json")
    out = ["% AUTO-GENERATED from pushback_readout_{nano,llama,grok,pooled}.json (17.9; haiku unread, guard).",
           "\\begin{tabular}{lrrrr}", "\\toprule",
           "model & items & capitulation CoT / NoT & re-ask CoT / NoT & net NoT $-$ CoT \\\\", "\\midrule"]

    def fmt(v):
        return "--" if not v else f"{v['cot']:.2f} / {v['not']:.2f}"

    def net(v):
        if not v or not v.get("not_minus_cot"):
            return "--"
        x = v["not_minus_cot"]
        return f"${100 * x['diff']:+.1f}$ $[{100 * x['lo']:+.1f}, {100 * x['hi']:+.1f}]$"
    for lab, n, cap, inst, nt in rows:
        out.append(f"{lab} & {n} & {fmt(cap)} & {fmt(inst)} & {net(nt)} \\\\")
    if pooled:
        p = pooled["pooled"]
        out.append("\\midrule")
        out.append(f"pooled & {p['cap']['not_minus_cot']['n_pairs']} & {fmt(p.get('cap'))} & {fmt(p.get('inst'))} & {net(p.get('net'))} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def seat_table() -> str:
    d = _load(OUT / "seat_scaffold_comparison.json")
    if d is None:
        return "% seat_scaffold_comparison.json not present\n"
    out = ["% AUTO-GENERATED from seat_scaffold_comparison.json (16.28; narrated minus plain-CoT seats, paired items).",
           "\\begin{tabular}{lrr}", "\\toprule", "readout & grok & Llama \\\\", "\\midrule"]
    names = [("acc", "collective accuracy"), ("routed", "routed accuracy"),
             ("routed_matched", "routed, matched coverage"), ("gain", "routing gain"), ("fire", "fire rate")]
    for k, lab in names:
        cells = []
        for m in ("grok", "llama"):
            v = d["per_model"][m][k]
            cells.append(f"${100 * v['diff']:+.1f}$ $[{100 * v['lo']:+.1f}, {100 * v['hi']:+.1f}]$")
        out.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


RUN_HEADS = {"1": "run 1 (original)", "2": "run 2 (same day, 2,048)", "3": "run 3 (same day, 4,096)"}


def _run_cell(c):
    return "" if not c or c.get("drop") is None else \
        f"${100 * c['drop']:+.1f}$ $[{100 * c['lo']:+.1f}, {100 * c['hi']:+.1f}]$"


def runs_table() -> str:
    """Appendix tab-app-runs: NoT minus CoT drop in each generation run and pooled over runs, one block
    per judge (panel order), one row per generator; run 2 exists on haiku, grok, nano and Llama only.
    Source run_stability_readout.json (the earlier scratch readout, NOT pre-registered), kept only
    while the appendix still inputs figures/tab_runs; the promoted artefact is run_stability.json,
    read by run_stability_table() into figures/tab_run_stability.tex (same per-run values, pooled
    values within 1.8 points)."""
    d = _load(OUT / "run_stability_readout.json")
    if d is None:
        return "% run_stability_readout.json not present\n"
    order = [jk for jk, *_ in JUDGES]
    judges = [j for j in order if j in d["judges"]] + [j for j in d["judges"] if j not in order]
    runs = sorted(d["runs"])
    out = ["% AUTO-GENERATED from run_stability_readout.json (scripts/analyze_run_stability.py; added after the",
           "% runs, not pre-registered). Points, 95 percent item bootstrap (8,000 draws, seed 20260822); pooled",
           "% resamples items, run a fixed crossed factor. Run 2 exists on haiku, grok, nano and Llama only.",
           "% SUPERSEDED SOURCE: the promoted artefact is run_stability.json (figures/tab_run_stability.tex).",
           "\\begin{tabular}{l" + "r" * (len(runs) + 1) + "}", "\\toprule",
           "generator & " + " & ".join(RUN_HEADS.get(r, f"run {r}") for r in runs) + " & pooled over runs \\\\"]
    for j in judges:
        out.append("\\midrule")
        out.append(f"\\multicolumn{{{len(runs) + 2}}}{{l}}{{judge {JUDGE_SHORT.get(j, j)}}} \\\\")
        for gk, gl in GENS:
            v = d["judges"][j]["per_generator"].get(gk)
            if not v or not v.get("per_run"):
                continue
            cells = [_run_cell(v["per_run"].get(r)) for r in runs] + [_run_cell(v["pooled"])]
            ns = ", ".join(f"n{r}={v['per_run'][r]['n_items']}" for r in runs if r in v["per_run"])
            out.append(f"{gl} & " + " & ".join(cells) + f" \\\\  % {ns}, pooled items {v['pooled']['n_items']}")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def run_stability_table() -> str:
    """figures/tab_run_stability.tex from run_stability.json (scripts/analyze_run_stability.py, post hoc,
    zero spend): NoT minus CoT per run, pooled with run as a fixed crossed factor, pooled with run as a
    random factor (method-of-moments variance components, normal interval), and the smallest pairwise
    item-level correlation of the drop across runs. One block per judge in panel order."""
    d = _load(OUT / "run_stability.json")
    if d is None:
        return "% run_stability.json not present\n"
    order = [jk for jk, *_ in JUDGES]
    judges = [j for j in order if j in d["judges"]] + [j for j in d["judges"] if j not in order]
    runs = sorted(d["runs"])
    ncol = len(runs) + 4
    out = ["% AUTO-GENERATED by scripts/make_iclr_figures.py run_stability_table from run_stability.json",
           "% (scripts/analyze_run_stability.py; added after the runs, NOT pre-registered; prereg block",
           "% 'Zero-spend post hoc analyses for review round 1'). Points; per-run and fixed-run pooled",
           "% intervals are 95 percent item-clustered percentile bootstrap (8,000 draws, seed 20260822),",
           "% the random-run pooled interval is normal with the run variance component from a two-way",
           "% method-of-moments fit (judges.<judge>.per_generator.<gen>.pooled_random_run; 1 df for run on",
           "% two-run generators, component clipped at zero where negative). item r = min over run pairs of",
           "% the item-level Pearson r of the drop (stability.min_pairwise_drop_r). Run 2 exists on haiku,",
           "% grok, nano and Llama only. Runs: " + "; ".join(f"{r}: {d['runs'][r]['label']}" for r in runs),
           "\\begin{tabular}{l" + "r" * (ncol - 1) + "}", "\\toprule",
           "generator & " + " & ".join(RUN_HEADS.get(r, f"run {r}") for r in runs)
           + " & pooled, run fixed & pooled, run random & item $r$ \\\\"]
    for j in judges:
        s = d["summary"]["per_judge"].get(j, {})
        out.append("\\midrule")
        out.append(f"\\multicolumn{{{ncol}}}{{l}}{{judge {JUDGE_SHORT.get(j, j)}}} \\\\"
                   f"  % summary.per_judge: all runs negative excl. 0 {s.get('n_all_runs_negative_ci_excl_0')} of "
                   f"{s.get('n_generators')}, same sign all runs {s.get('n_same_sign_all_runs')}, sign flips "
                   f"{s.get('generators_sign_flip')}, pooled negative excl. 0 {s.get('n_pooled_negative_ci_excl_0')}, "
                   f"max range {s.get('max_range_points', float('nan')):.1f} pts ({s.get('generator_max_range')}), "
                   f"mean item r {s.get('mean_item_drop_r', float('nan')):.2f}, mean run SD "
                   f"{s.get('mean_sigma_run_points', float('nan')):.1f} pts")
        for gk, gl in GENS:
            v = d["judges"][j]["per_generator"].get(gk)
            if not v or not v.get("per_run"):
                continue
            rr = v.get("pooled_random_run") or {}
            cells = [_run_cell(v["per_run"].get(r)) for r in runs] + [_run_cell(v["pooled"]), _run_cell(rr)]
            st = v.get("stability", {})
            r_min = st.get("min_pairwise_drop_r")
            cells.append("" if r_min is None else f"{r_min:.2f}")
            ns = ", ".join(f"n{r}={v['per_run'][r]['n_items']}" for r in runs if r in v["per_run"])
            out.append(f"{gl} & " + " & ".join(cells) + f" \\\\  % {ns}, pooled items {v['pooled']['n_items']}, "
                       f"range {v['pooled'].get('range_points', float('nan')):.1f} pts, run SD "
                       f"{rr.get('sigma_run_points', float('nan')):.1f} pts, same sign {st.get('same_sign_all_runs')}")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


Z3_MODELS = [("llama", "Llama-3.3-70B-Instruct", "Llama"), ("mistral", "Mistral-Large-3-2", "Mistral"),
             ("deepseek", "DeepSeek-V4-Pro", "DeepSeek")]
Z3_PROMPTS = [("standard", "CoT"), ("narrative_cot", "NoT")]


def z3_table() -> str:
    """figures/tab_z3.tex from z3_sdt_{llama,mistral,deepseek}.json codings.published. Six rows model x
    prompt; criterion c in each framing, the PERSON shift with its 95 percent item-bootstrap interval,
    d' in each framing, hit and false-alarm rates in each framing."""
    missing = []
    out = ["% AUTO-GENERATED by scripts/make_iclr_figures.py z3_table from z3_sdt_<model>.json (analysis block Z3),",
           "% codings.published: '<model>|<prompt>|third_person' and '|as_asker' (criterion, d_prime, hit_rate,",
           "% fa_rate) and decomposition '<model>|<prompt>|PERSON (3rd -> as_asker)' (delta, ci_low, ci_high).",
           "% CoT here is the panel's unscaffolded four-sentence prompt (scaffold key standard), not Figure 1's CoT.",
           "\\begin{tabular}{llrrrrrrr}", "\\toprule",
           "model & prompt & $c$ third & $c$ writer & shift $[95\\%]$ & $d'$ third & $d'$ writer & hit / FA third & hit / FA writer \\\\",
           "\\midrule"]
    for short, full, label in Z3_MODELS:
        z = _load(OUT / f"z3_sdt_{short}.json")
        if z is None:
            missing.append(f"z3_sdt_{short}.json")
            continue
        d = z["codings"]["published"]
        for pk, pl in Z3_PROMPTS:
            a = d[f"{full}|{pk}|third_person"]
            b = d[f"{full}|{pk}|as_asker"]
            s = d["decomposition"][f"{full}|{pk}|PERSON (3rd -> as_asker)"]
            out.append(f"{label} & {pl} & ${a['criterion']:+.2f}$ & ${b['criterion']:+.2f}$ & "
                       f"${s['delta']:+.2f}$ $[{s['ci_low']:+.2f}, {s['ci_high']:+.2f}]$ & "
                       f"{a['d_prime']:.2f} & {b['d_prime']:.2f} & "
                       f"{a['hit_rate']:.2f} / {a['fa_rate']:.2f} & {b['hit_rate']:.2f} / {b['fa_rate']:.2f} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    out.insert(4, f"% missing at generation time: {missing or 'none'}")
    return "\n".join(out) + "\n"


def interjudge_table() -> str:
    """figures/tab_interjudge.tex from interjudge_kappa.json (scripts/analyze_interjudge_kappa.py, post hoc,
    zero spend): each judge's validation rate over the 2,004 scored OEQ CoT and NoT responses, agreement and
    Cohen's kappa with the production judge (all responses, CoT only, NoT only; item-clustered bootstrap,
    8,000 draws, seed 20260822) and the kappa ceiling the two marginals allow. First row, when
    judge_test_retest.json is present: the production judge against itself on the 814 responses whose
    judge input is byte-identical between the production scorer and the 17.8 full-text pass."""
    d = _load(OUT / "interjudge_kappa.json")
    if d is None:
        return "% interjudge_kappa.json not present\n"
    t = _load(OUT / "judge_test_retest.json")
    prod = d["production_judge"]
    order = [jk for jk, *_ in JUDGES if jk != prod]
    judges = [j for j in order if j in d["kappa_vs_production"]] + \
             [j for j in d["kappa_vs_production"] if j not in order]
    prod_rate = d["base_rate_region"][prod]["rate_all"]
    ref = d.get("reference_human_answers", {})
    out = ["% AUTO-GENERATED by scripts/make_iclr_figures.py interjudge_table from interjudge_kappa.json",
           "% (scripts/analyze_interjudge_kappa.py; post hoc, zero spend; prereg block 'Zero-spend post hoc",
           "% analyses for review round 1'). rate = kappa_vs_production.<judge>.all.rate_1 (the judge's",
           f"% validation rate over all scored responses; production judge {prod_rate:.3f} = base_rate_region",
           "% rate_all); agreement and kappa [lo, hi] = kappa_vs_production.<judge>.{all,standard_cot,",
           "% narrative_cot}.{agreement,kappa,kappa_lo,kappa_hi}; kappa_max = .all.kappa_max_given_marginals.",
           f"% Human reference rate {ref.get('reference_rate', float('nan')):.3f} on {ref.get('n_items')} items",
           "% (judge_reference_agreement.json). Test-retest row: judge_test_retest.json pooled and by_arm",
           "% (scripts/analyze_judge_test_retest.py; draw 1 = production scorer, draw 2 = 17.8 full-text pass,",
           "% same model, rubric and max_tokens, responses of at most 4,000 characters).",
           f"% judge_test_retest.json present: {t is not None}",
           "\\begin{tabular}{lrrrrrr}", "\\toprule",
           "judge & rate & agreement & $\\kappa$, all & $\\kappa$, CoT & $\\kappa$, NoT & $\\kappa_{\\max}$ \\\\",
           "\\midrule"]

    def kci(c, k="kappa", lo="kappa_lo", hi="kappa_hi"):
        return f"{c[k]:.3f} [{c[lo]:.3f}, {c[hi]:.3f}]"

    def tci(c):
        ci = c.get("kappa_ci95") or [float("nan"), float("nan")]
        return f"{c['kappa']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
    if t is not None:
        p = t["pooled"]
        ba = t.get("by_arm", {})
        cot = tci(ba["standard_cot"]) if "standard_cot" in ba else "--"
        nott = tci(ba["narrative_cot"]) if "narrative_cot" in ba else "--"
        out.append(f"{JUDGE_SHORT.get(prod, prod)}, test-retest & {prod_rate:.3f} & {p['agreement']:.3f} & "
                   f"{tci(p)} & {cot} & {nott} & -- \\\\"
                   f"  % n {p['n']} identical-input pairs, {p['n_items']} items; rates draw 1 {p['rate_draw1']:.3f}, "
                   f"draw 2 {p['rate_draw2']:.3f}; CoT n {ba.get('standard_cot', {}).get('n')}, NoT n "
                   f"{ba.get('narrative_cot', {}).get('n')}")
        out.append("\\midrule")
    for j in judges:
        k = d["kappa_vs_production"][j]
        a = k["all"]
        out.append(f"{JUDGE_SHORT.get(j, j)} & {a['rate_1']:.3f} & {a['agreement']:.3f} & {kci(a)} & "
                   f"{kci(k['standard_cot'])} & {kci(k['narrative_cot'])} & {a['kappa_max_given_marginals']:.3f} \\\\"
                   f"  % n {a['n']} (CoT {k['standard_cot']['n']}, NoT {k['narrative_cot']['n']}), items {a['n_items']}")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


KNOCK_GENS = [("claude-haiku-4-5", "haiku"), ("grok-4-1-fast-reasoning", "grok"), ("gpt-5.4-nano", "nano")]
KNOCK_ARMS = [("standard_cot", "CoT"), ("narrative_cot", "intact NoT"),
              ("not_drop_stakeholders", "NoT without stakeholders"),
              ("not_drop_consequences", "NoT without consequences"),
              ("not_drop_uncertainty", "NoT without uncertainty"),
              ("not_commit_first", "NoT, decision first")]


def knockout_table() -> str:
    """figures/tab_knockout_readouts.tex from readouts_17_3_17_5.json knockouts_17_3.per_generator
    (scripts/analyze_readouts_17_3_17_5.py, post hoc, zero spend): per generator and condition the
    non-response share of the 150 items, compliance (share of non-empty responses carrying every
    mandated section label), mean length in characters, the condition's validation rate minus intact
    NoT's on the items scored in both (points) and the Manski bracket on that difference (every
    unscored item imputed first as not validating, then as validating; printed as one number where
    nothing is missing). CoT and intact NoT rows carry the descriptives only."""
    d = _load(OUT / "readouts_17_3_17_5.json")
    if d is None:
        return "% readouts_17_3_17_5.json not present\n"
    k = d["knockouts_17_3"]["per_generator"]
    out = ["% AUTO-GENERATED by scripts/make_iclr_figures.py knockout_table from readouts_17_3_17_5.json",
           "% (scripts/analyze_readouts_17_3_17_5.py; added after review, NOT pre-registered; prereg block",
           "% 'Zero-spend post hoc analyses for review round 1'). knockouts_17_3.per_generator.<gen>.<arm>:",
           "% nonresponse_pct, compliance_pct, mean_len_chars, knock_minus_intact_pp.complete_case and",
           "% .manski_lo / .manski_hi (points; every unscored item of 150 imputed first as not validating,",
           "% then as validating; a single number where lo = hi). Scores are the production judge's full-text",
           "% re-scores; grok's and nano's CoT and intact NoT are the June cells, the 17.3 comparators.",
           "\\begin{tabular}{llrrrrr}", "\\toprule",
           "generator & condition & non-response & compliance & mean chars & minus intact NoT & bracket \\\\",
           "\\midrule"]
    for gi, (gk, gl) in enumerate(KNOCK_GENS):
        g = k.get(gk, {})
        first = True
        for ak, al in KNOCK_ARMS:
            a = g.get(ak)
            if not a:
                continue
            comp = "--" if a.get("compliance_pct") is None else f"{a['compliance_pct']:.1f}\\%"
            kmi = a.get("knock_minus_intact_pp")
            if kmi:
                diff = f"${kmi['complete_case']:+.1f}$"
                lo, hi = kmi["manski_lo"], kmi["manski_hi"]
                br = f"${lo:+.1f}$" if abs(hi - lo) < 0.05 else f"$[{lo:+.1f}, {hi:+.1f}]$"
            else:
                diff, br = "", ""
            out.append(f"{gl if first else ''} & {al} & {a['nonresponse_pct']:.1f}\\% & {comp} & "
                       f"{a['mean_len_chars']:,.0f} & {diff} & {br} \\\\"
                       f"  % n_nonempty {a['n_nonempty']}, n_scored {a['n_scored']}, rate "
                       f"{a['rate_pct_complete_case']:.1f}")
            first = False
        if gi < len(KNOCK_GENS) - 1:
            out.append("\\addlinespace")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    outputs = {"fig_judge_panel.tex": judge_panel_figure(), "tab_routing.tex": routing_table(),
               "tab_form.tex": form_table(), "tab_form_main.tex": form_main_table(),
               "tab_rep4k.tex": rep4k_table(), "tab_pushback.tex": pushback_table(),
               "tab_seat.tex": seat_table(), "tab_runs.tex": runs_table(),
               "tab_run_stability.tex": run_stability_table(), "tab_z3.tex": z3_table(),
               "tab_interjudge.tex": interjudge_table(),
               "tab_knockout_readouts.tex": knockout_table()}
    for name, text in outputs.items():
        (FIG / name).write_text(text)
        print("wrote", FIG / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
