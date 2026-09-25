"""
scripts/make_iclr_figures.py -- figure and table sources for papers/iclr2027, generated from
the committed artefacts only (every number carries a % source comment in the output).

  figures/fig_judge_panel.tex    NOT WRITTEN by main() since 2026-09-24 (Figure 2 is being redesigned by
                                 scripts/iclr_fig_judge_heatmap.py). Formerly: pillar 1: NoT minus CoT validation drop per generator under
                                 every judge, two panels. Left, the 17.10 same-day regeneration
                                 of both prompts at 4,096 tokens (headline_rep4k_readout.json,
                                 the registered headline); right, the original responses of the
                                 17.8 panel (judge_panel_full_oeq_<judge>.json)
  figures/tab_routing.tex        pillar 3: collective, routed, routing gain and the sensor's
                                 precision and recall per condition (verify_pillar3_headline_*.json),
                                 the same model's solo, and the paired collective minus solo and
                                 routed collective minus solo (collective_vs_solo_paired.json)
  figures/tab_form_main.tex      main-text table, prompts as rows, five validation judges and two
                                 framing judges as columns, points only with a circle where the
                                 interval includes zero (narrative_form_readout.json,
                                 baselines_17_12_readout.json, framing_17_11_readout.json)
  figures/tab_form.tex           appendix form table, validation blocks (five judges) and accepting-
                                 framing blocks (two judges), readings and Holm daggers
                                 (narrative_form_readout.json, framing_17_11_readout.json, multiplicity_mde.json)
  figures/tab_baselines.tex      appendix, the four prompting baselines of 17.12 on both rubrics
                                 (baselines_17_12_readout.json, framing_17_11_readout.json key baselines)
  figures/tab_human_anchor.tex   appendix, every prompt minus the judge-scored human rate on both rubrics
                                 (human_anchor_readout.json, 17.11 amendment 2)
  figures/tab_rep4k.tex          17.10 replicate by judge with Holm daggers, run 3b and pooled rows for
                                 sonnet and haiku, and a framing panel (headline_rep4k_readout.json,
                                 headline_rep4k_b_readout.json, run_stability.json, framing_17_11_readout.json,
                                 judge_panel_full_rep4k_framing.json, multiplicity_mde.json)
  figures/tab_pushback.tex       17.9 capitulation (pushback_readout_*.json)
  figures/tab_seat.tex           16.28 narrated minus plain seats (seat_scaffold_comparison.json)
  figures/tab_runs.tex           run-to-run drops from the earlier scratch readout
                                 (run_stability_readout.json; kept while the appendix inputs it)
  figures/tab_run_stability.tex  run-to-run drops from the promoted artefact (run_stability.json),
                                 production judge and gpt-4o; figures/tab_run_stability_b.tex the other three
  figures/tab_z3.tex             asker shielding, signal detection (z3_sdt_<model>.json)
  figures/tab_interjudge.tex     inter-judge kappa against the production judge
  figures/tab_knockout_readouts.tex  17.3 knockout descriptives and Manski brackets (readouts_17_3_17_5.json)
  figures/fig_collective_prompts.tex  every prompt of the collective, its routing judge and the
                                 validation judge, printed verbatim from the scripts (W25)
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
# Row labels (clarity round 2, 2026-09-24, collective-05): the edges-off condition is called the exchange cut
# in the paper, so its row reads "grok, exchange cut" and the primary row "grok (primary)".
CONDITIONS = [("grok", "grok (primary)"), ("noedge", "grok, exchange cut"),
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
    Marks are 2.3pt (2.5pt touched neighbouring rows at the earlier 5.0cm height; 3.5cm since 2026-09-23 for the page budget) with a per-judge row offset."""
    style = {jk: (col, mark, fill) for jk, _jl, col, mark, fill in JUDGES}
    k = len(series)
    opts = [f"name={name}", "scale only axis", f"title={{{title}}}",
            "title style={font=\\small, yshift=-3pt}", "width=0.40\\columnwidth", "height=3.5cm",
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
           "% grok +2.4 [+0.5, +4.5], haiku +3.1 [+1.2, +5.1]). Exchange cut (edges off) has no solo pairing (blank).",
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


JUDGE_SHORT = {"claude-haiku-4-5": "haiku (production)", "gpt-4o": "gpt-4o", "gpt-5.4-nano": "nano",
               "Llama-3.3-70B-Instruct": "Llama", "grok-4-1-fast-reasoning": "grok"}


def _ci(c, scale=100):
    # Interval in math mode so its minus signs match the point estimate's (appendix tables).
    return f"${scale * c['point']:+.1f}$ $[{scale * c['lo']:+.1f}, {scale * c['hi']:+.1f}]$"


# ---- legibility helpers (2026-09-24, senior author: no table body below \footnotesize, no \resizebox
# shrinking a table below 0.95 of its natural size). Every generated table sets its own size inside a
# group, so the section wrapper needs only \centering and \input. Where inline intervals would not fit
# the 397.5pt text width, each estimate takes two table rows, the point on the first and its interval
# on the second (LEG_OPEN / LEG_CLOSE and the _two_row helper).
LEG_OPEN = "{\\footnotesize"
LEG_CLOSE = "}"
ZERO_MARK = "\\rlap{$^{\\circ}$}"   # main-text table: the 95 percent interval includes zero


def _pp(x, scale=100):
    """A point estimate in points, math mode, signed."""
    return f"${scale * x:+.1f}$"


def _iv(lo, hi, scale=100):
    """A 95 percent interval in points, math mode, signed."""
    return f"$[{scale * lo:+.1f}, {scale * hi:+.1f}]$"


def _norm(c):
    """(point, lo, hi) from either artefact convention: {point, lo, hi}, {drop, lo, hi} or {diff, lo, hi}."""
    if c is None:
        return None
    for k in ("point", "drop", "diff"):
        if k in c and c[k] is not None:
            return c[k], c["lo"], c["hi"]
    return None


def _incl0(c) -> bool:
    p = _norm(c)
    return p is not None and p[1] <= 0 <= p[2]


def _two_row(label1, label2, cells, extra1="", extra2=""):
    """Two table rows for one estimate per cell: the points (with label1 in the stub) and, below,
    the intervals (with label2 in the stub). cells holds (point, lo, hi, mark) tuples or None."""
    r1 = [label1] + ["--" if c is None else _pp(c[0]) + (c[3] if len(c) > 3 else "") for c in cells]
    r2 = [label2] + ["" if c is None else _iv(c[1], c[2]) for c in cells]
    return [" & ".join(r1) + " \\\\" + extra1, " & ".join(r2) + " \\\\[1pt]" + extra2]


def _judge_head(stub, judges):
    """Two header rows naming the judges, with "(production)" on the second row under the production
    judge, so that the column is no wider than its intervals (one-line "haiku (production)" is 67pt at
    \\footnotesize against 56.5pt for an interval)."""
    prod = JUDGES[0][0]
    r1 = f"{stub} & " + " & ".join(j_short(j) for j in judges) + " \\\\"
    r2 = " & " + " & ".join("(production)" if j == prod else "" for j in judges) + " \\\\"
    return [r1, r2]


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
# the NoT-minus-prompt column (gap) has its own families for the checklist and narrative-only contrasts
FORM_GAP_FAMILY = {"checklist": "checklist_gap", "narrative_only": "narrative_only_gap"}
# The two judges of the accepting-framing rubric (prereg 17.11), in panel order.
FRAMING_JUDGES = ["claude-haiku-4-5", "gpt-4o"]
# The four prompting baselines of prereg 17.12 (scripts/scaffold_permutations.BASELINES and the
# third-person rewrite of scripts/make_third_person_oeq.py), in the registration's order.
BASELINES = [("cot_prepend_naive", "naive instruction"), ("cot_prepend_context", "context instruction"),
             ("cot_perspective_shift", "third-person rewrite"), ("cot_sceptical", "sceptical persona")]


def _judges_in(d_judges):
    order = [jk for jk, *_ in JUDGES]
    return [j for j in order if j in d_judges] + [j for j in d_judges if j not in order]


def form_table() -> str:
    """Appendix tab-app-form: the form, content and persona tests, pooled over haiku, grok, nano and
    Llama, one block of rows per judge, first judged validation under the five judges
    (narrative_form_readout.json, prereg 17.7, 17.7 amendment 2, 17.4) and then acceptance of the asker's
    framing under the production judge and gpt-4o on the same responses (framing_17_11_readout.json
    design (b), prereg 17.11). Columns: the prompt's change from same-run CoT, NoT minus the prompt,
    that change as a share of NoT's, and the registered reading (validation only; the framing rubric's
    single reading, FRAMING-MIXED, is over the whole pattern and goes in the caption)."""
    d = _load(OUT / "narrative_form_readout.json")
    if d is None:
        return "% narrative_form_readout.json not present\n"
    fr = _load(OUT / "framing_17_11_readout.json")
    holm = _load(OUT / "multiplicity_mde.json")
    fam = holm["tables"]["table6_form"]["per_judge"] if holm else None
    out = ["% AUTO-GENERATED by scripts/make_iclr_figures.py form_table. Validation blocks from",
           "% narrative_form_readout.json (prereg 17.7, 17.7 amendment 2, 17.4; same-run replicates standard_cot_rep /",
           "% narrative_cot_rep as comparators; pooled over haiku, grok, nano, Llama; item-clustered bootstrap):",
           "% change = form_pooled.<arm>.drop_arm (NoT: form_pooled.checklist.drop_not; persona: persona_pooled),",
           "% NoT - prompt = .gap (drop_not - drop_arm), share = .share.point, reading = form_reading / persona_reading.",
           "% Framing blocks from framing_17_11_readout.json judges.<judge>.{not,checklist,narrative_only,persona}_minus_cot",
           "% (prereg Addendum 17.11 RESULTS design (b); the same responses re-scored with the accepting-framing",
           "% rubric by the production judge and gpt-4o; 8,000 draws, seed 20260822; registered reading",
           f"% {fr.get('reading') if fr else 'n/a'} over both judges, stated in the caption).",
           "% Dagger: multiplicity_mde.json tables.table6_form.per_judge.<judge>.<family>.cells, families",
           "% not_rep_drop / checklist_drop / narrative_only_drop / persona_drop (change column) and checklist_gap /",
           "% narrative_only_gap (NoT - prompt column) over the four generators within the judge (m = 4, exact p).",
           "% The rows here are POOLED and the artefact tests each generator, so the dagger marks a pooled cell",
           "% whose family has at least one generator with a raw interval excluding zero that the Holm step-down",
           "% does not keep; the per-row comment gives survivors of m and the generators removed. Framing cells",
           "% carry no dagger (no multiplicity artefact covers them).",
           f"% multiplicity_mde.json present: {holm is not None}; framing_17_11_readout.json present: {fr is not None}",
           LEG_OPEN, "\\setlength{\\tabcolsep}{4pt}",
           "\\begin{tabular}{@{}lrrrl@{}}", "\\toprule",
           "prompt & change from CoT & NoT $-$ prompt & share of NoT & reading \\\\", "\\midrule"]
    any_dagger = False

    def mark(judge, family):
        nonlocal any_dagger
        if not family or not fam or judge not in fam or family not in fam[judge]:
            return "", ""
        f = fam[judge][family]
        removed = [gl for gk, gl in GENS if _fails_holm(f["cells"].get(gk))]
        note = (f" Holm {family}: {f['n_survive_holm']} of {f['m']} survive, raw excluding zero "
                f"{f['n_raw_ci_excludes_0']}" + (f", removed {removed}" if removed else "") + ";")
        if removed:
            any_dagger = True
            return DAGGER, note
        return "", note

    first = True
    for j in _judges_in(d["judges"]):
        jr = d["judges"][j]
        f = jr["form_pooled"]
        ck, no, pc = f["checklist"], f["narrative_only"], jr.get("persona_pooled", {})
        if ck.get("drop_arm") is None:
            continue
        if not first:
            out.append("\\addlinespace")
        first = False
        out.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{judged validation, {JUDGE_SHORT.get(j, j)} judge}}}} \\\\")
        dg, nt = mark(j, FORM_FAMILY["NoT"])
        out.append(f"NoT & {_ci(ck['drop_not'])}{dg} & & $1.00$ & \\\\  %{nt} pairs {ck.get('n_pairs')}")
        for key, lab, blk, reading in (("checklist", "checklist", ck, jr["form_reading"]["checklist"]),
                                       ("narrative_only", "narrative only", no, jr["form_reading"]["narrative_only"]),
                                       ("persona", "persona only", pc, jr.get("persona_reading", ""))):
            if blk.get("drop_arm") is None:
                continue
            dg, nt = mark(j, FORM_FAMILY[key])
            dg2, nt2 = mark(j, FORM_GAP_FAMILY.get(key))
            out.append(f"{lab} & {_ci(blk['drop_arm'])}{dg} & {_ci(blk['gap'])}{dg2} & "
                       f"${blk['share']['point']:.2f}$ & {reading} \\\\  %{nt}{nt2} pairs {blk.get('n_pairs')}")
    if fr is not None:
        for j in [jk for jk in FRAMING_JUDGES if jk in fr["judges"]]:
            jr = fr["judges"][j]
            out.append("\\addlinespace")
            out.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{accepting the asker's framing, "
                       f"{JUDGE_SHORT.get(j, j)} judge}}}} \\\\")
            nm = jr["not_minus_cot"]
            out.append(f"NoT & {_ci(nm['drop_arm'])} & & $1.00$ & \\\\  % not_minus_cot, pairs {nm['n_pairs']}")
            for key, lab in (("checklist_minus_cot", "checklist"),
                             ("narrative_only_minus_cot", "narrative only"), ("persona_minus_cot", "persona only")):
                b = jr[key]
                out.append(f"{lab} & {_ci(b['drop_arm'])} & {_ci(b['gap'])} & ${b['share']['point']:.2f}$ & \\\\"
                           f"  % {key}, pairs {b['n_pairs']}, share [{b['share']['lo']:.2f}, {b['share']['hi']:.2f}],"
                           f" NoT's drop on these pairs {100 * b['drop_not']['point']:+.1f}")
    # The dagger is explained in the caption (a footnote row set in \\multicolumn{5}{l} would widen the
    # last column to the length of its sentence).
    out += ["\\bottomrule", f"% daggers present: {any_dagger}", "\\end{tabular}", LEG_CLOSE]
    return "\n".join(out) + "\n"


def baselines_table() -> str:
    """Appendix tab-app-baselines (new, 2026-09-24): the four prompting baselines of prereg 17.12,
    pooled over haiku, grok, nano and Llama, one block per judge, first judged validation under the
    five judges (baselines_17_12_readout.json, prereg 17.12 RESULTS) and then acceptance of the asker's
    framing under the production judge and gpt-4o on the same responses (framing_17_11_readout.json key
    baselines, prereg 17.11 amendment RESULTS). Columns: the baseline's change from CoT, NoT minus the
    baseline and the checklist minus the baseline, each paired on items; the registered readings follow
    mechanically from the last two intervals (checked here against the artefacts' readings)."""
    bl = _load(OUT / "baselines_17_12_readout.json")
    fr = _load(OUT / "framing_17_11_readout.json")
    frb = fr.get("baselines") if fr else None
    if bl is None and frb is None:
        return "% baselines_17_12_readout.json and framing_17_11_readout.json key baselines not present\n"
    out = ["% AUTO-GENERATED by scripts/make_iclr_figures.py baselines_table. Validation blocks from",
           "% baselines_17_12_readout.json judges.<judge>.pooled.<baseline>.{not,checklist} (prereg Addendum 17.12",
           "% RESULTS; paired item-clustered bootstrap, 8,000 draws, seed 20260923): change = not.drop_arm, NoT -",
           "% baseline = not.gap (drop_not - drop_arm), checklist - baseline = checklist.gap. Framing blocks from",
           "% framing_17_11_readout.json baselines.judges.<judge>.pooled (prereg Addendum 17.11 amendment RESULTS;",
           "% 8,000 draws, seed 20260822), same fields. Comparators standard_cot_rep, narrative_cot_rep and",
           "% not_checklist of 22 and 23 September, one to two days before the baselines, same 2,048 cap (nano",
           "% 8,192). Readings (prereg 17.12, applied to the framing rate by the 17.11 amendment): BASELINE-BEATS-X",
           "% if the X - baseline interval lies above 0, X-BEATS-BASELINE if below, MATCH if it includes 0; each",
           "% row's comment gives the artefact's readings, checked against that rule.",
           f"% baselines_17_12_readout.json present: {bl is not None}; framing_17_11_readout.json baselines present: {frb is not None}",
           LEG_OPEN, "\\setlength{\\tabcolsep}{4pt}",
           "\\begin{tabular}{lrrr}", "\\toprule",
           "baseline & change from CoT & NoT $-$ baseline & checklist $-$ baseline \\\\", "\\midrule"]
    mismatches = []
    started = [False]

    def rule(c, comp):
        lo, hi = c["lo"], c["hi"]
        return f"BASELINE-BEATS-{comp}" if lo > 0 else (f"{comp}-BEATS-BASELINE" if hi < 0 else "MATCH")

    def block(src, judges, rubric):
        for j in judges:
            jr = src["judges"][j]
            if started[0]:
                out.append("\\addlinespace")
            started[0] = True
            out.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\textit{{{rubric}, {JUDGE_SHORT.get(j, j)} judge}}}} \\\\")
            for bk, bl_ in BASELINES:
                p = jr["pooled"].get(bk)
                if not p:
                    continue
                n, c = p["not"], p["checklist"]
                rd = jr.get("readings", {}).get(bk, {})
                for comp, cell, key in (("NOT", n["gap"], "not"), ("CHECKLIST", c["gap"], "checklist")):
                    if rd.get(key) and rd.get(key) != rule(cell, comp):
                        mismatches.append(f"{rubric} {j} {bk} {key}: artefact {rd.get(key)} vs rule {rule(cell, comp)}")
                out.append(f"{bl_} & {_ci(n['drop_arm'])} & {_ci(n['gap'])} & {_ci(c['gap'])} \\\\"
                           f"  % readings vs NoT {rd.get('not')}, vs checklist {rd.get('checklist')}; pairs "
                           f"{n['n_pairs']} / {c['n_pairs']}; NoT's drop {100 * n['drop_not']['point']:+.1f}, "
                           f"checklist's drop {100 * c['drop_not']['point']:+.1f}")
    if bl is not None:
        block(bl, _judges_in(bl["judges"]), "judged validation")
    if frb is not None:
        block(frb, [j for j in FRAMING_JUDGES if j in frb["judges"]], "accepting the asker's framing")
    out += ["\\bottomrule", "\\end{tabular}", LEG_CLOSE]
    out.insert(out.index(LEG_OPEN), f"% reading/rule mismatches: {mismatches or 'none'}")
    return "\n".join(out) + "\n"


def rep4k_table() -> str:
    """Appendix tab-app-rep4k, two panels. (a) Judged validation, NoT minus CoT per generator under the
    five judges in the 4,096-token regeneration (headline_rep4k_readout.json, prereg 17.10), each
    estimate on two rows (point, then interval) with the generator's pre-declared reading under its
    name, and below the seven generators the second 4,096-token run on sonnet and haiku
    (headline_rep4k_b_readout.json, prereg 17.10 amendment run 2 = run 3b of run_stability.json) and
    the drop pooled over every run of that generator (run_stability.json pooled, run fixed). (b)
    Acceptance of the asker's framing on the same responses under the production judge and gpt-4o
    (framing_17_11_readout.json design (a), paired rates from judge_panel_full_rep4k_framing.json;
    prereg 17.11 RESULTS)."""
    d = _load(OUT / "headline_rep4k_readout.json")
    if d is None:
        return "% headline_rep4k_readout.json not present\n"
    b = _load(OUT / "headline_rep4k_b_readout.json")
    rs = _load(OUT / "run_stability.json")
    fr = _load(OUT / "framing_17_11_readout.json")
    fp = _load(OUT / "judge_panel_full_rep4k_framing.json")
    holm = _holm_cells("table5_rep4k")
    judges = _judges_in(d["judges"])
    nj = len(judges)
    out = ["% AUTO-GENERATED by scripts/make_iclr_figures.py rep4k_table. Panel (a): headline_rep4k_readout.json",
           "% per_generator.<gen>.per_judge.<judge> (drop, lo, hi) and .reading (prefix HEADLINE- dropped) (prereg",
           "% 17.10 RESULTS: same-day, both arms at 4,096 tokens, nano 8,192). Run 3b rows: headline_rep4k_b_readout.json",
           "% per_generator.<gen>.per_judge (prereg 'Addendum 17.10 amendment, run 2 RESULTS'; its own readings are",
           "% not printed, the registration fixed that the 17.10 readings are not re-read); pooled rows:",
           "% run_stability.json judges.<judge>.per_generator.<gen>.pooled (run fixed, equal run weights, items",
           "% resampled; sonnet runs 1, 3, 3b, haiku runs 1, 2, 3, 3b; not pre-registered, replaces the registered",
           "% two-run pool, deviation stated in the appendix). Dagger: multiplicity_mde.json tables.table5_rep4k",
           "% .per_judge.<judge>.cells.<gen> with ci_excludes_0 true and survives_holm_exact false (Holm step-down",
           "% over the seven generators within the judge, exact item-bootstrap p).",
           "% Panel (b): framing_17_11_readout.json judges.<judge>.design_a_per_generator.<gen>.drop_arm and",
           "% design_a_not_minus_cot_pooled (prereg Addendum 17.11 RESULTS design (a)); CoT and NoT rates on the",
           "% paired items from judge_panel_full_rep4k_framing.json judges.<judge>.per_generator.<gen>",
           "% .narrative_cot_rep4k.{rate_a, rate_b} (the RESULTS block quotes the rates over all scored",
           "% responses, which differ by at most 1 point on nano, grok, Llama, Mistral and DeepSeek).",
           f"% present: multiplicity_mde.json {holm is not None}, headline_rep4k_b_readout.json {b is not None}, "
           f"run_stability.json {rs is not None}, framing_17_11_readout.json {fr is not None}, "
           f"judge_panel_full_rep4k_framing.json {fp is not None}",
           LEG_OPEN, "\\setlength{\\tabcolsep}{4pt}",
           "\\textit{(a) Judged validation, NoT $-$ CoT}\\\\[2pt]",
           "\\begin{tabular}{@{}l" + "c" * nj + "@{}}", "\\toprule"] + _judge_head("generator", judges) + ["\\midrule"]
    daggered = []
    for gk, gl in GENS:
        v = d["per_generator"].get(gk)
        if not v:
            continue
        cells = []
        for j in judges:
            p = _norm(v["per_judge"].get(j))
            if p is None:
                cells.append(None)
                continue
            dg = ""
            if holm and _fails_holm(holm.get(j, {}).get(gk)):
                dg = "\\rlap{" + DAGGER + "}"
                daggered.append(f"{JUDGE_SHORT.get(j, j)} judge x {gl} (adj. p {holm[j][gk]['holm_adj_p_exact']:.3f})")
            cells.append((*p, dg))
        # The reading is stated in the caption (a stub wide enough for JUDGE-DEPENDENT, 84.5pt, would push the
        # table past the text width); it stays in this row's comment.
        out += _two_row(gl, "", cells,
                        extra1=f"  % reading {v['reading'].replace('HEADLINE-', '')}; n_items "
                               f"{', '.join(str(v['per_judge'][j].get('n_items')) for j in judges if j in v['per_judge'])}")
    extra = []
    for gk, gl in (("claude-sonnet-4-6", "sonnet"), ("claude-haiku-4-5", "haiku")):
        if b is not None and b["per_generator"].get(gk):
            bv = b["per_generator"][gk]["per_judge"]
            cells = [None if _norm(bv.get(j)) is None else _norm(bv.get(j)) for j in judges]
            if rs is not None:   # cross-check against run_stability.json run 3b
                for j in judges:
                    r3b = rs["judges"].get(j, {}).get("per_generator", {}).get(gk, {}).get("per_run", {}).get("3b")
                    if r3b and bv.get(j) and abs(r3b["drop"] - bv[j]["drop"]) > 1e-9:
                        extra.append(f"% WARNING run 3b mismatch {gl} {j}")
            extra += _two_row(f"{gl}, run 3b", "", cells,
                              extra1=f"  % headline_rep4k_b_readout.json per_generator.{gk}.per_judge (artefact reading "
                                     f"{b['per_generator'][gk].get('reading')}, not printed)")
        if rs is not None:
            pc = [_norm(rs["judges"].get(j, {}).get("per_generator", {}).get(gk, {}).get("pooled")) for j in judges]
            runs = rs["judges"][judges[0]]["per_generator"][gk]["pooled"].get("runs")
            extra += _two_row(f"{gl}, pooled", "", pc,
                              extra1=f"  % run_stability.json judges.<judge>.per_generator.{gk}.pooled, runs {runs}")
    if extra:
        out.append("\\midrule")
        out += extra
    # The dagger is explained in the caption, not in a footnote row, which would widen the last column.
    out += ["\\bottomrule", f"% daggered (explained in the caption): {'; '.join(daggered) or 'none'}", "\\end{tabular}"]
    if fr is not None:
        fj = [j for j in FRAMING_JUDGES if j in fr["judges"]]
        out += ["", "\\medskip", "\\textit{(b) Accepting the asker's framing, CoT and NoT rates and NoT $-$ CoT}\\\\[2pt]",
                "\\begin{tabular}{@{}l" + "rrr" * len(fj) + "@{}}", "\\toprule",
                " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{JUDGE_SHORT.get(j, j)} judge}}" for j in fj) + " \\\\",
                "".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(fj))),
                "generator & " + " & ".join("CoT & NoT & NoT $-$ CoT" for _ in fj) + " \\\\", "\\midrule"]
        for gk, gl in GENS:
            cells, notes = [], []
            for j in fj:
                c = fr["judges"][j]["design_a_per_generator"].get(gk)
                pr = (fp or {}).get("judges", {}).get(j, {}).get("per_generator", {}).get(gk, {}).get("narrative_cot_rep4k")
                if not c:
                    cells += ["--", "--", "--"]
                    continue
                if pr and abs(pr["drop"] - c["drop_arm"]["point"]) > 1e-9:
                    notes.append(f"WARNING drop mismatch {j}")
                ra = f"{pr['rate_a']:.3f}" if pr else "--"
                rb = f"{pr['rate_b']:.3f}" if pr else "--"
                cells += [ra, rb, _ci(c["drop_arm"])]
                notes.append(f"{JUDGE_SHORT.get(j, j)} n {c['n_pairs']}")
            out.append(f"{gl} & " + " & ".join(cells) + f" \\\\  % {'; '.join(notes)}")
        out.append("\\midrule")
        pooled = [fr["judges"][j]["design_a_not_minus_cot_pooled"] for j in fj]
        out.append("pooled & " + " & ".join(f"& & {_ci(p['drop_arm'])}" for p in pooled) + " \\\\"
                   f"  % design_a_not_minus_cot_pooled, pairs {', '.join(str(p['n_pairs']) for p in pooled)}")
        out += ["\\bottomrule", "\\end{tabular}"]
    out += [LEG_CLOSE]
    return "\n".join(out) + "\n"


def form_main_table() -> str:
    """Main-text tab-form-main: prompts as rows, judges as columns, five for judged validation
    (narrative_form_readout.json) and two for acceptance of the asker's framing
    (framing_17_11_readout.json), pooled over haiku, grok, nano and Llama. Points only, one row per
    prompt so that the table is legible at \\footnotesize within the text width; a circle marks an
    entry whose 95 percent interval includes zero, and every interval is in the row's comment and in
    the appendix table tab-app-form (the length control is in tab-app-length, the prompting baselines
    of prereg 17.12 in tab-app-baselines)."""
    d = _load(OUT / "narrative_form_readout.json")
    if d is None:
        return "% narrative_form_readout.json not present\n"
    fr = _load(OUT / "framing_17_11_readout.json")
    vj = _judges_in(d["judges"])
    fj = [j for j in FRAMING_JUDGES if fr is not None and j in fr["judges"]]
    out = ["% AUTO-GENERATED by scripts/make_iclr_figures.py form_main_table. Points; pooled over haiku, grok, nano",
           "% and Llama, item-clustered bootstrap; each row's comment gives every entry with its 95 percent interval.",
           "% Validation (five judges): narrative_form_readout.json judges.<judge>.form_pooled.<arm>.{drop_arm, drop_not,",
           "% gap}, persona_pooled.drop_arm (prereg 17.7 and 17.7 amendment 2 RESULTS).",
           "% Framing (production judge, gpt-4o): framing_17_11_readout.json judges.<judge>.<arm>_minus_cot.{drop_arm,",
           "% gap} (prereg 17.11 RESULTS design (b)). gap = NoT's drop minus the arm's drop on the same pairs = NoT minus",
           "% the prompt. Circle: the 95 percent interval includes zero. The length-matched rows are in tab-app-length",
           "% and the four prompting baselines (prereg 17.12, generated one to two days after CoT) in tab-app-baselines.",
           f"% present: framing_17_11_readout.json {fr is not None}",
           LEG_OPEN, "\\setlength{\\tabcolsep}{4.5pt}",
           "\\begin{tabular}{l" + "r" * len(vj) + ("r" * len(fj)) + "}", "\\toprule",
           f" & \\multicolumn{{{len(vj)}}}{{c}}{{judged validation}}"
           + (f" & \\multicolumn{{{len(fj)}}}{{c}}{{accepting the framing}}" if fj else "") + " \\\\",
           f"\\cmidrule(lr){{2-{1 + len(vj)}}}" + (f"\\cmidrule(lr){{{2 + len(vj)}-{1 + len(vj) + len(fj)}}}" if fj else ""),
           "points & " + " & ".join(j_short(j) for j in vj + fj) + " \\\\", "\\midrule"]

    def row(label, vpick, fpick):
        cells, notes = [], []
        for j, src, pick in [(j, d["judges"][j], vpick) for j in vj] + \
                            [(j, fr["judges"][j] if fr else None, fpick) for j in fj]:
            c = None if (pick is None or src is None) else pick(j, src)
            p = _norm(c)
            if p is None:
                cells.append("--")
                continue
            cells.append(_pp(p[0]) + (ZERO_MARK if p[1] <= 0 <= p[2] else ""))
            notes.append(f"{j_short(j)} {100 * p[0]:+.1f} [{100 * p[1]:+.1f}, {100 * p[2]:+.1f}]")
        out.append(f"{label} & " + " & ".join(cells) + " \\\\  % " + "; ".join(notes))

    fp = lambda key, fld: (lambda j, jr: jr[key][fld])  # noqa: E731  framing pick
    row("NoT", lambda j, jr: jr["form_pooled"]["checklist"].get("drop_not"), fp("not_minus_cot", "drop_arm"))
    row("checklist", lambda j, jr: jr["form_pooled"]["checklist"].get("drop_arm"), fp("checklist_minus_cot", "drop_arm"))
    row("narrative only", lambda j, jr: jr["form_pooled"]["narrative_only"].get("drop_arm"),
        fp("narrative_only_minus_cot", "drop_arm"))
    row("persona only", lambda j, jr: jr.get("persona_pooled", {}).get("drop_arm"), fp("persona_minus_cot", "drop_arm"))
    out.append("\\midrule")
    row("NoT $-$ checklist", lambda j, jr: jr["form_pooled"]["checklist"].get("gap"), fp("checklist_minus_cot", "gap"))
    row("NoT $-$ narrative only", lambda j, jr: jr["form_pooled"]["narrative_only"].get("gap"),
        fp("narrative_only_minus_cot", "gap"))
    # The four baseline rows (prereg 17.12 and the 17.11 amendment; naive instruction, context instruction,
    # third-person rewrite, sceptical persona) left this table on 2026-09-24 (pillar1 fix pass) for the
    # main-text page budget. They were generated one to two days after the CoT they are subtracted from
    # (prereg 17.12 Comparators), so they were also the only rows of this table that changed date; they are
    # in the appendix table tab-app-baselines with their intervals and that date caveat in its caption, and
    # Section sec-single points there.
    # The length-matched rows (NoT minus length-matched CoT on haiku and grok) left this table on 2026-09-24
    # to pay for the framing columns and baseline rows in the main-text page budget; they are in the
    # appendix table tab-app-length with their intervals (narrative_form_readout.json length_per_generator).
    out += ["\\bottomrule", "\\end{tabular}", LEG_CLOSE]
    return "\n".join(out) + "\n"


def j_short(j):
    """Column heads of the main-text table: judge names without the production marker, which the caption gives."""
    return {"claude-haiku-4-5": "haiku", "Llama-3.3-70B-Instruct": "Llama", "grok-4-1-fast-reasoning": "grok",
            "gpt-5.4-nano": "nano"}.get(j, j)


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
           "model & items & capitulation CoT / NoT & re-ask instability CoT / NoT & net NoT $-$ CoT \\\\", "\\midrule"]

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
    """Appendix tab-app-seat: narrated against plain chain-of-thought seats, per model, the levels of each
    collective beside the paired difference (seat_scaffold_comparison.json per_model). grok and Llama
    are the registered pair (prereg 16.28); haiku, when the artefact holds it, is the pair of prereg
    16.28 amendment E-D, registered later and read on its own."""
    d = _load(OUT / "seat_scaffold_comparison.json")
    if d is None:
        return "% seat_scaffold_comparison.json not present\n"
    # Levels of each collective beside the paired difference (clarity loop 2, 2026-09-24, m-seats-05), so the
    # accuracies the main text points to are in the table; the difference cells are unchanged.
    models = [("grok", "grok"), ("llama", "Llama")] + ([("haiku", "haiku")] if "haiku" in d["per_model"] else [])
    pmr = d.get("per_model_reading", {})
    out = ["% AUTO-GENERATED by scripts/make_iclr_figures.py seat_table from seat_scaffold_comparison.json",
           "% (prereg Addendum 16.28, grok and Llama, the registered pair; haiku from prereg 'Addendum 16.28",
           "% amendment E-D RESULTS', registered later and read on its own, on the debates both haiku cells",
           "% complete under the 16.23 partial-panel rule). per_model.<m>.<readout>.{narrated, plain, diff, lo, hi};",
           "% per-item means of narrated and plain seats and their paired difference (scripts/analyze_seat_scaffold.py,",
           "% 8,000 draws, seed 20260922). Readings per model: "
           + "; ".join(f"{m} {pmr.get(m, {}).get('primary')}" for m, _ in models),
           "% Set at \\small (it fits the text width at that size; 254pt at \\footnotesize).",
           "{\\small", "\\setlength{\\tabcolsep}{4pt}",
           "\\begin{tabular}{lrrr}", "\\toprule", "readout & narrated & plain & narrated minus plain \\\\"]
    names = [("acc", "collective accuracy"), ("routed", "routed accuracy"),
             ("routed_matched", "routed, matched coverage"), ("gain", "routing gain"), ("fire", "fire rate")]
    for m, head in models:
        r = d["per_model"][m]
        comp = "; ".join(f"{c['tag']} {c['rows']} of {c['expected']}" + (f" ({c['partial_panel_rule']})"
                                                                          if c.get("partial_panel_rule") else "")
                         for c in r.get("completeness", []))
        out += ["\\midrule", f"\\multicolumn{{4}}{{@{{}}l}}{{\\textit{{{head}, {r['acc']['n_items']} items}}}} \\\\"
                f"  % completeness: {comp}"]
        for k, lab in names:
            v = r[k]
            out.append(f"{lab} & {v['narrated']:.3f} & {v['plain']:.3f} & "
                       f"${100 * v['diff']:+.1f}$ $[{100 * v['lo']:+.1f}, {100 * v['hi']:+.1f}]$ \\\\")
    out += ["\\bottomrule", "\\end{tabular}", LEG_CLOSE]
    return "\n".join(out) + "\n"


# Run names (clarity loop 2, 2026-09-24, appendix1-03): run 3 is "the 4,096-token regeneration", as in
# Section sec-single ("The regeneration without truncation") and Appendix app-rep4k, and run 2 is the CoT
# and NoT generated at 2,048 tokens with the form prompts, their same-run comparator in Appendix app-form
# (loop 1 had called runs 2 and 3 "the same-run replicate").
# Each head is stacked on two lines so that the longer names do not widen the resized table and shrink
# its type (measured 2026-09-24: one-line heads took the body from 5.35pt to 4.89pt).
RUN_HEADS = {"1": "\\shortstack[r]{run 1\\\\original}", "2": "\\shortstack[r]{run 2, 2,048\\\\form-test run}",
             "3": "\\shortstack[r]{run 3, 4,096\\\\regeneration}",
             "3b": "\\shortstack[r]{run 3b, 4,096\\\\second regeneration}"}


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


def run_stability_table(judge_subset=None) -> str:
    """figures/tab_run_stability.tex (production judge and gpt-4o) and figures/tab_run_stability_b.tex
    (nano, Llama and grok judges) from run_stability.json (scripts/analyze_run_stability.py, post hoc,
    zero spend, rerun with --include-extra so that run 3b, the second 4,096-token run of prereg 17.10
    amendment run 2, is included): NoT minus CoT per run, pooled with run as a fixed crossed factor,
    pooled with run as a random factor (method-of-moments variance components, normal interval), and the
    smallest pairwise item-level correlation of the drop across runs. One block per judge in panel order.
    Set at \\footnotesize for a sidewaystable (the eight columns with inline intervals run to about 615pt,
    more than the portrait text width holds even with each interval on a second row), and the 35
    generator-by-judge rows are split over two tables so that each fits a rotated page (2026-09-24,
    legibility)."""
    d = _load(OUT / "run_stability.json")
    if d is None:
        return "% run_stability.json not present\n"
    judges = _judges_in(d["judges"])
    if judge_subset is not None:
        judges = [j for j in judges if j in judge_subset]
    runs = sorted(d["runs"])
    heads = {"1": ("run 1", "original"), "2": ("run 2", "2,048"), "3": ("run 3", "4,096"), "3b": ("run 3b", "4,096")}
    ncol = len(runs) + 4
    out = ["% AUTO-GENERATED by scripts/make_iclr_figures.py run_stability_table from run_stability.json",
           "% (scripts/analyze_run_stability.py; added after the runs, NOT pre-registered; prereg block",
           "% 'Zero-spend post hoc analyses for review round 1'; rerun with --include-extra for run 3b, prereg",
           "% 'Addendum 17.10 amendment, run 2 RESULTS'). Points; per-run and fixed-run pooled",
           "% intervals are 95 percent item-clustered percentile bootstrap (8,000 draws, seed 20260822),",
           "% the random-run pooled interval is normal with the run variance component from a two-way",
           "% method-of-moments fit (judges.<judge>.per_generator.<gen>.pooled_random_run; 1 df for run on",
           "% two-run generators, component clipped at zero where negative). item r = min over run pairs of",
           "% the item-level Pearson r of the drop (stability.min_pairwise_drop_r). Run 2 exists on haiku,",
           "% grok, nano and Llama only, run 3b on haiku and sonnet only. Judges in this file: "
           + ", ".join(JUDGE_SHORT.get(j, j) for j in judges) + ".",
           "% Runs: " + "; ".join(f"{r}: {d['runs'][r]['label']}" for r in runs)
           + (" (3b = the second 4,096-token run, standard_cot_rep4k_b / narrative_cot_rep4k_b)" if "3b" in runs else ""),
           "% Set for a sidewaystable (one row per generator, intervals inline): 615pt wide at \\footnotesize,",
           "% which the 397.5pt portrait text width cannot hold even with intervals on a second row.",
           LEG_OPEN, "\\setlength{\\tabcolsep}{4pt}",
           "\\begin{tabular}{@{}l" + "r" * (ncol - 1) + "@{}}", "\\toprule",
           " & " + " & ".join(heads.get(r, (f"run {r}", ""))[0] for r in runs) + " & pooled & pooled & item \\\\",
           "generator & " + " & ".join(heads.get(r, ("", ""))[1] for r in runs) + " & run fixed & run random & $r$ \\\\"]
    for j in judges:
        s = d["summary"]["per_judge"].get(j, {})
        out.append("\\midrule")
        out.append(f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\textit{{judge {JUDGE_SHORT.get(j, j)}}}}} \\\\"
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
            # math mode, and -0.00 printed as 0.00 (clarity loop 2, appendix1-21)
            cells.append("" if r_min is None else f"${(0.0 if abs(r_min) < 0.005 else r_min):.2f}$")
            ns = ", ".join(f"n{r}={v['per_run'][r]['n_items']}" for r in runs if r in v["per_run"])
            out.append(f"{gl} & " + " & ".join(cells) + f" \\\\  % {ns}, pooled items {v['pooled']['n_items']}, "
                       f"range {v['pooled'].get('range_points', float('nan')):.1f} pts, run SD "
                       f"{rr.get('sigma_run_points', float('nan')):.1f} pts, same sign {st.get('same_sign_all_runs')}")
    out += ["\\bottomrule", "\\end{tabular}", LEG_CLOSE]
    return "\n".join(out) + "\n"


# Prompts of the human-anchor readout (prereg 17.11 amendment 2), in the order of Table tab-form-main.
ANCHOR_ARMS = [("standard_cot_rep", "CoT"), ("narrative_cot_rep", "NoT"), ("not_checklist", "checklist"),
               ("not_narrative_only", "narrative only"), ("advisor_cot", "persona only")] + BASELINES
ANCHOR_WORD = {"ABOVE-HUMAN": "above", "AT-HUMAN": "at", "BELOW-HUMAN": "below"}


def human_anchor_table() -> str:
    """Appendix tab-app-human-anchor (new, 2026-09-24): every prompt's rate minus the same judge's rate on
    the human answers to the same items (ELEPHANT's own definition of social sycophancy, rate minus the
    human rate, a rate below the human rate being what ELEPHANT calls over-correction), pooled over haiku,
    grok, nano and Llama, per judge, on both rubrics, with the pre-declared reading
    (human_anchor_readout.json, prereg 17.11 amendment 2 RESULTS; descriptive). Panel (a) judged
    validation under the five judges, panel (b) acceptance of the asker's framing under the production
    judge and gpt-4o. In panel (a) each entry takes two rows, its point with the reading word and below
    it its interval; panel (b), two columns, fits with the interval on the same row. The first row of
    each panel is the judge's rate on the human answers."""
    d = _load(OUT / "human_anchor_readout.json")
    if d is None:
        return "% human_anchor_readout.json not present\n"
    mism = []
    out = ["% AUTO-GENERATED by scripts/make_iclr_figures.py human_anchor_table from human_anchor_readout.json",
           "% (scripts/analyze_human_anchor.py; prereg 'Addendum 17.11 amendment 2' registered 2026-09-23 before the",
           "% framing calls on the human answers, descriptive; RESULTS in 'Addendum 17.11 amendment RESULTS and",
           "% amendment 2 RESULTS'). rubrics.<rubric>.<judge>.human_rate (the judge's own rate on the human answers,",
           "% n_human answers), .pooled.<arm>.{diff, lo, hi} (arm rate minus human rate on the same items, pooled over",
           "% the four 17.7 generators, item-clustered paired bootstrap, 8,000 draws, seed 20260923) and",
           "% .readings.<arm> (ABOVE-HUMAN interval above 0, BELOW-HUMAN below 0, AT-HUMAN otherwise; printed as",
           "% above / below / at and checked against that rule). Human answers' framing scores from",
           "% judge_reference_agreement_framing.json, validation scores from judge_reference_agreement.json (17.8",
           "% amendment 2). CoT is standard_cot_rep (the 2,048-token CoT of the form tests); persona only is advisor_cot.",
           LEG_OPEN, "\\setlength{\\tabcolsep}{3.5pt}"]
    panels = [("validation", "(a) Judged validation, prompt $-$ human answers", None),
              ("framing", "(b) Accepting the asker's framing, prompt $-$ human answers", FRAMING_JUDGES)]
    for pi, (rub, title, only) in enumerate(panels):
        R = d["rubrics"].get(rub)
        if not R:
            out.append(f"% rubric {rub} not present")
            continue
        js = _judges_in(R) if only is None else [j for j in only if j in R]
        if pi:
            out += ["", "\\medskip"]
        out += [f"\\textit{{{title}}}\\\\[2pt]",
                "\\begin{tabular}{@{}l" + ("rl" if len(js) <= 2 else "c") * len(js) + "@{}}", "\\toprule"] \
            + (_judge_head("prompt", js) if len(js) > 2 else
               ["prompt & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{JUDGE_SHORT.get(j, j)}}}" for j in js) + " \\\\"]) \
            + ["\\midrule",
                "human rate & " + " & ".join((f"\\multicolumn{{2}}{{c}}{{${R[j]['human_rate']:.3f}$}}" if len(js) <= 2
                                              else f"${R[j]['human_rate']:.3f}$") for j in js) + " \\\\"
                f"  % human_rate; n_human {', '.join(str(R[j]['n_human']) for j in js)}", "\\midrule"]
        inline = len(js) <= 2   # a two-judge panel fits with the interval on the same row
        for ak, al in ANCHOR_ARMS:
            cells1, cells2, notes = [], [], []
            for j in js:
                c = R[j]["pooled"].get(ak)
                if not c:
                    cells1.append("-- & " if inline else "--")
                    cells2.append("")
                    continue
                rd = R[j]["readings"].get(ak)
                rule = "ABOVE-HUMAN" if c["lo"] > 0 else ("BELOW-HUMAN" if c["hi"] < 0 else "AT-HUMAN")
                if rd != rule:
                    mism.append(f"{rub} {j} {ak}: {rd} vs {rule}")
                if inline:   # the reading word in a left-aligned column of its own, so the intervals align
                    cells1.append(f"{_pp(c['diff'])} {_iv(c['lo'], c['hi'])} & {ANCHOR_WORD.get(rd, rd)}")
                else:
                    cells1.append(f"{_pp(c['diff'])} {ANCHOR_WORD.get(rd, rd)}")
                cells2.append(_iv(c["lo"], c["hi"]))
                notes.append(f"{JUDGE_SHORT.get(j, j)} {rd} pairs {c['n_pairs']} items {c['n_items']}")
            out.append(f"{al} & " + " & ".join(cells1) + " \\\\  % " + "; ".join(notes))
            if not inline:
                out.append(" & " + " & ".join(cells2) + " \\\\[1pt]")
        out += ["\\bottomrule", "\\end{tabular}"]
    out += [LEG_CLOSE]
    out.insert(10, f"% reading/rule mismatches: {mism or 'none'}")
    return "\n".join(out) + "\n"


# The four generators of the form tests (prereg 17.7), in the order of their per-generator readouts.
FORM_GENS = [("claude-haiku-4-5", "haiku"), ("grok-4-1-fast-reasoning", "grok"), ("gpt-5.4-nano", "nano"),
             ("Llama-3.3-70B-Instruct", "Llama")]


def framing_gen_table() -> str:
    """Appendix tab-app-framing-gen (appendix editor, 2026-09-24): acceptance of the asker's framing on
    the form-test responses, per generator, under the production judge and gpt-4o
    (framing_17_11_readout.json judges.<judge>.design_b_per_generator; prereg 17.11 RESULTS design (b)).
    Four changes from same-run CoT (NoT, checklist, narrative only, persona only) and two gaps (NoT minus
    the checklist, NoT minus narrative only), each estimate on two rows, point then interval. The pooled
    contrasts are in tab_form.tex and are not repeated here."""
    fr = _load(OUT / "framing_17_11_readout.json")
    if fr is None:
        return "% framing_17_11_readout.json not present\n"
    fj = [j for j in FRAMING_JUDGES if j in fr["judges"]]
    arms = [("not", "drop_arm"), ("checklist", "drop_arm"), ("narrative_only", "drop_arm"), ("persona", "drop_arm"),
            ("checklist", "gap"), ("narrative_only", "gap")]
    out = ["% AUTO-GENERATED by scripts/make_iclr_figures.py framing_gen_table from framing_17_11_readout.json",
           "% judges.<judge>.design_b_per_generator.<gen>.<arm>.{drop_arm, gap} (prereg Addendum 17.11 RESULTS,",
           "% design (b): the 17.7 responses of standard_cot_rep, narrative_cot_rep, not_checklist,",
           "% not_narrative_only and advisor_cot re-scored with ELEPHANT's accepting-framing rubric; paired",
           "% item bootstrap, 8,000 draws, seed 20260822). drop_arm = the prompt's rate minus same-run CoT's on",
           "% the items scored under CoT, NoT and the prompt; gap = NoT's drop minus the prompt's on those items.",
           "% Rates are shares of answers the judge scores as accepting the asker's framing (1), not challenging",
           "% its premise. No reading is registered per generator (the registered reading, FRAMING-MIXED, is on the",
           "% pooled contrasts of tab_form.tex).",
           LEG_OPEN, "\\setlength{\\tabcolsep}{2.5pt}",   # 3pt ran 2.4pt past the 397.5pt text width
           "\\begin{tabular}{@{}lcccccc@{}}", "\\toprule",
           " & \\multicolumn{4}{c}{change from same-run CoT} & \\multicolumn{2}{c}{NoT $-$ prompt} \\\\",
           "\\cmidrule(lr){2-5}\\cmidrule(lr){6-7}",
           "generator & NoT & checklist & narrative only & persona only & checklist & narrative only \\\\",
           "\\midrule"]
    for ji, j in enumerate(fj):
        if ji:
            out.append("\\addlinespace")
        out.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{JUDGE_SHORT.get(j, j)} judge}}}} \\\\")
        pg = fr["judges"][j].get("design_b_per_generator", {})
        for gk, gl in FORM_GENS:
            v = pg.get(gk)
            if not v:
                continue
            cells, pairs = [], []
            for ak, field in arms:
                c = v.get(ak)
                p = _norm(c.get(field)) if c else None
                cells.append(p)
                if c and field == "drop_arm":
                    pairs.append(f"{ak} {c['n_pairs']}")
            out += _two_row(gl, "", cells, extra1=f"  % pairs {', '.join(pairs)}")
    out += ["\\bottomrule", "\\end{tabular}", LEG_CLOSE]
    return "\n".join(out) + "\n"


def _tex_prompt(text: str) -> str:
    """A prompt string typeset verbatim in the body font: TeX specials escaped, straight double quotes
    turned into opening and closing quotes in order, curly quotes into TeX quotes, the em dash into ---,
    line breaks kept (a blank line becomes a small vertical gap). Used for the prompt figures below,
    whose text cannot go through the verbatim environment because the build's typewriter font (OT1
    cmtt) has no em dash or curly quotes."""
    rep = {"\\": "\\textbackslash{}", "&": "\\&", "%": "\\%", "$": "\\$", "#": "\\#", "_": "\\_",
           "{": "\\{", "}": "\\}", "~": "\\textasciitilde{}", "^": "\\textasciicircum{}",
           "<": "\\textless{}", ">": "\\textgreater{}", "—": "---", "–": "--",
           "“": "``", "”": "''", "‘": "`", "’": "'"}
    out, open_q = [], True
    for ch in text:
        if ch == '"':
            out.append("``" if open_q else "''")
            open_q = not open_q
        else:
            out.append(rep.get(ch, ch))
    s = "".join(out).strip("\n")
    paras = [p for p in s.split("\n\n")]
    return "\\\\[3pt]\n".join("\\\\\n".join(ln if ln.strip() else "~" for ln in p.split("\n")) for p in paras)


def framing_rubric_figure() -> str:
    """figures/fig_framing_rubric.tex (appendix editor, 2026-09-24): the accepting-framing judge's system
    and user turns, printed from scripts/elephant_scorers.py (SCORER_SYSTEM, _framing_prompt, the verbatim
    port of ELEPHANT's framing scorer) with <QUESTION> and <RESPONSE> standing for the item and the full
    response. The body of a boxed minipage; the float and caption are in sections/appendix.tex."""
    from scripts.elephant_scorers import SCORER_SYSTEM, _framing_prompt
    return "\n".join([
        "% AUTO-GENERATED by scripts/make_iclr_figures.py framing_rubric_figure from scripts/elephant_scorers.py",
        "% SCORER_SYSTEM and _framing_prompt('<QUESTION>', '<RESPONSE>') (verbatim port of ELEPHANT's",
        "% sycophancy_scorers.py framing prompt; prereg Addendum 17.11 'Path'). Straight double quotes are set",
        "% as TeX quotes and the em dash as ---; every other character is the module's.",
        "\\textbf{System turn.} " + _tex_prompt(SCORER_SYSTEM) + "\\\\[5pt]",
        "\\textbf{User turn.}\\\\",
        _tex_prompt(_framing_prompt("<QUESTION>", "<RESPONSE>")),
    ]) + "\n"


def baseline_prompts_figure() -> str:
    """figures/fig_baseline_prompts.tex (appendix editor, 2026-09-24): the prompting baselines of prereg
    17.12, printed from scripts/scaffold_permutations.BASELINES (the three system prompts) and
    scripts/make_third_person_oeq.ELEPHANT_REWRITE_PROMPT (the rewriter's instruction). The body of a
    boxed minipage; the float and caption are in sections/appendix.tex."""
    from scripts.scaffold_permutations import BASELINES as BP
    from scripts.make_third_person_oeq import ELEPHANT_REWRITE_PROMPT, REWRITER
    return "\n".join([
        "% AUTO-GENERATED by scripts/make_iclr_figures.py baseline_prompts_figure from",
        "% scripts/scaffold_permutations.BASELINES (cot_prepend_naive, cot_prepend_context, cot_sceptical; the",
        "% first two end with ELEPHANT's sentences, Table A13 of arXiv 2505.13995v2 as transcribed in prereg",
        "% 17.12) and scripts/make_third_person_oeq.ELEPHANT_REWRITE_PROMPT (ELEPHANT Appendix G.2, curly",
        f"% quotation marks of the PDF set as TeX quotes; rewriter {REWRITER}, empty system prompt, the",
        "% instruction, a blank line and the question as the user turn).",
        "\\textbf{Naive instruction.} " + _tex_prompt(BP["cot_prepend_naive"]) + "\\\\[5pt]",
        "\\textbf{Context instruction.} " + _tex_prompt(BP["cot_prepend_context"]) + "\\\\[5pt]",
        "\\textbf{Sceptical persona.} " + _tex_prompt(BP["cot_sceptical"]) + "\\\\[5pt]",
        "\\textbf{Third-person rewrite, the rewriter's instruction.} " + _tex_prompt(ELEPHANT_REWRITE_PROMPT),
    ]) + "\n"


Z3_MODELS = [("llama", "Llama-3.3-70B-Instruct", "Llama"), ("mistral", "Mistral-Large-3-2", "Mistral"),
             ("deepseek", "DeepSeek-V4-Pro", "DeepSeek")]
Z3_PROMPTS = [("standard", "brief"), ("narrative_cot", "NoT")]


def z3_table() -> str:
    """figures/tab_z3.tex from z3_sdt_{llama,mistral,deepseek}.json codings.published. Six rows model x
    prompt; criterion c in each framing, the PERSON shift with its 95 percent item-bootstrap interval,
    d' in each framing, hit and false-alarm rates in each framing."""
    missing = []
    out = ["% AUTO-GENERATED by scripts/make_iclr_figures.py z3_table from z3_sdt_<model>.json (analysis block Z3),",
           "% codings.published: '<model>|<prompt>|third_person' and '|as_asker' (criterion, d_prime, hit_rate,",
           "% fa_rate) and decomposition '<model>|<prompt>|PERSON (3rd -> as_asker)' (delta, ci_low, ci_high).",
           "% brief is the unscaffolded four-sentence prompt used on the dispute set (scaffold key standard), not Figure 1's CoT.",
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
           "% re-scores; every generator's CoT and intact NoT are the June cells, the 17.3 comparators.",
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



def collective_prompts_figure() -> str:
    """figures/fig_collective_prompts.tex: every prompt the collective of Section 6 and its routing
    judge use, printed verbatim from the modules that ran them (scripts/run_crowdgold_deliberation.py,
    scripts/run_crowdgold_aita.py, scripts/verify_pillar3_headline.py, scripts/elephant_scorers.py),
    with <POST TEXT> standing for the account and <...> for prior-round text. Reviewer W25."""
    import inspect
    import textwrap
    from scripts import run_crowdgold_deliberation as m
    from scripts.run_crowdgold_aita import build_system, build_user_turn
    from scripts.elephant_scorers import SCORER_SYSTEM, _validation_prompt
    sig = inspect.signature(m.run).parameters.get("allow_unresolved")
    allow = True if sig is None or sig.default is inspect._empty else bool(sig.default)
    item = m.CrowdGoldItem(item_id="<item>", post_text="<POST TEXT>", gold_verdict="YTA")
    prior = {rid: "<THAT SEAT'S PRIOR STATEMENT>" for rid in m.ROLE_ORDER}
    labels = {rid: "<THAT SEAT'S LABEL>" for rid in m.ROLE_ORDER}
    objections = {rid: "<THAT SEAT'S STATED MODIFICATION>" for rid in m.ROLE_ORDER}
    writer = m.ROLE_BY_ID["writer_advocate"]

    def tex(t: str) -> str:
        """Escape a title or note for text mode (the prompts themselves go through verbatim)."""
        return (t.replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("&", "\\&")
                 .replace("%", "\\%").replace("#", "\\#").replace("<", "\\textless{}")
                 .replace(">", "\\textgreater{}"))

    shared = [(m.request_block("as_asker", "<POST TEXT>"), "<REQUEST WRAPPER, printed above>"),
              (m.GLOSSARY_CORE, "<GLOSSARY, printed above>"),
              (m._role_header(writer), "<ROLE BRIEF, printed above>")]

    def verb(text: str) -> str:
        for full, short in shared:
            if text != full:
                text = text.replace(full, short)
        lines = []
        for raw in text.rstrip("\n").split("\n"):
            lines.extend(textwrap.wrap(raw, 78, break_long_words=False, break_on_hyphens=False,
                                       subsequent_indent="  ") or [""])
        return "\\begin{verbatim}\n" + "\n".join(lines) + "\n\\end{verbatim}\n"

    blocks = [
        ("Seat system prompt", "The NoT text of Figure~FIGPROMPT, verbatim and alone, or the CoT "
         "text for the plain-seat collectives (agent_system).", None),
        ("Request wrapper, third person (build_user_turn)", None, build_user_turn("third_person", "<POST TEXT>")),
        ("Request wrapper, the user as the writer, quoted between the request markers (request_block)", None,
         m.request_block("as_asker", "<POST TEXT>")),
        ("Preamble shown to every seat (R0_PREAMBLE)", None, m.R0_PREAMBLE),
        ("Verdict-label glossary (GLOSSARY_CORE)", None, m.GLOSSARY_CORE),
        ("Forced verdict line, seats and moderator (verdict_instruction)", None,
         m.verdict_instruction(m.INSTRUMENT, allow_unresolved=allow)),
        ("Forced label line, stage 5", None, m.verdict_instruction(m.LABEL_INSTRUMENT, allow_unresolved=False)),
        ("Forced vote line, stage 7", None, m.verdict_instruction(m.VOTE_INSTRUMENT, allow_unresolved=False)),
    ]
    for r in m.ROLES:
        blocks.append((f"Role brief, {r.label} (_role_header)", None, m._role_header(r)))
    blocks += [
        ("Stage 1, opening statement, user turn (r0_user), shown for the writer's advocate", None,
         m.r0_user("as_asker", item, writer, allow_unresolved=allow)),
        ("Stage 2, rebuttal (r1_user)", None,
         m.r1_user("as_asker", item, writer, "<OWN OPENING>", prior, allow_unresolved=allow, cap=0)),
        ("Stage 3, final position (r2_user)", None,
         m.r2_user("as_asker", item, writer, "<OWN OPENING>", "<OWN REBUTTAL>", prior,
                   allow_unresolved=allow, cap=0)),
        ("Moderator system prompt, stage 4 (SYNTHESIS_SYSTEM)", None, m.SYNTHESIS_SYSTEM),
        ("Stage 4, synthesis, moderator user turn (synthesis_user)", None,
         m.synthesis_user("as_asker", item, prior, allow_unresolved=allow, cap=0)),
        ("Stage 5, label on the synthesis (r3_label_user)", None,
         m.r3_label_user("as_asker", item, writer, "<OWN FINAL POSITION>", "<SYNTHESIS>", cap=0)),
        ("Moderator system prompt, stage 6 (INTEGRATION_SYSTEM)", None, m.INTEGRATION_SYSTEM),
        ("Stage 6, integrated verdict, moderator user turn (integration_user)", None,
         m.integration_user("as_asker", item, "<SYNTHESIS>", labels, objections,
                            allow_unresolved=allow, cap=0)),
        ("Stage 7, vote (r4_vote_user)", None,
         m.r4_vote_user("as_asker", item, writer, "<SYNTHESIS>", "ACCEPT_WITH_MODIFICATION",
                        "<OWN STATED MODIFICATION>", "<INTEGRATED PROPOSAL>", cap=0)),
        ("Marker lines parsed from the seats' and moderator's text", None,
         "\n".join([m.MOD_MARKER, m.UNRESOLVABLE_MARKER, m.ADDRESSED_MARKER,
                    "SYNTHESIS:", "PROPOSAL:", "VERDICT: <LABEL>"])),
        # scaffold-04 (clarity round 3, 2026-09-24): was "Routing judge (sonnet), system prompt (...,
        # scaffold standard)"; "scaffold standard" read as the Standard chain-of-thought prompt of Figure
        # fig:prompt, and Section sec-scaffold points here for the solo's prompt too (Appendix
        # app-collective "Pairing with the solo"); the next note's semicolon is now ", and" (style rule)
        ("Brief unscaffolded prompt with neither NoT nor CoT, the system prompt of the solo and of the "
         "routing judge, sonnet (run_crowdgold_aita.build_system)",
         None, build_system(scaffold="standard", allow_unresolved=allow)),
        ("Routing judge, user turn", "The request wrapper above for the debate's framing, with the "
         "same account, answered three times, and the judge's verdict is the majority of the three "
         "(YTA and ESH coded at fault, NTA and NAH not) and a tie keeps the collective's verdict "
         "(verify_pillar3_headline.load and routed_ok).", None),
        ("Validation judge of Section~SECSINGLE, system prompt (SCORER_SYSTEM)", None, SCORER_SYSTEM),
        ("Validation judge, user turn (_validation_prompt), the full response in place of <RESPONSE>",
         None, _validation_prompt("<QUESTION>", "<RESPONSE>")),
    ]
    out = ["% AUTO-GENERATED by scripts/make_iclr_figures.py collective_prompts_figure (reviewer W25):",
           "% printed from scripts/run_crowdgold_deliberation.py (ROLES, R0_PREAMBLE, GLOSSARY_CORE,",
           "% verdict_instruction, _role_header, r0_user, r1_user, r2_user, synthesis_user, r3_label_user,",
           "% integration_user, r4_vote_user, SYNTHESIS_SYSTEM, INTEGRATION_SYSTEM, MOD_MARKER,",
           "% UNRESOLVABLE_MARKER, ADDRESSED_MARKER), scripts/run_crowdgold_aita.py (build_user_turn,",
           "% build_system with the standard scaffold and BREVITY), scripts/verify_pillar3_headline.py",
           "% (majority of three, ties to the collective) and scripts/elephant_scorers.py (SCORER_SYSTEM,",
           "% _validation_prompt). Stand-in text in angle brackets replaces the account and prior-round",
           f"% text; transcript cap 0 (no truncation of prior rounds); allow_unresolved={allow} as in run().",
           "% The stage prompts repeat the request wrapper, the glossary and the role brief verbatim; each",
           "% repetition is replaced here by a placeholder naming the block printed above, so that every",
           "% line of every prompt appears exactly once."]
    for title, note, text in blocks:
        out.append(f"\\noindent\\textbf{{{tex(title).replace('SECSINGLE', chr(92) + 'ref{sec-single}')}.}}")
        if note:
            out.append(tex(note).replace("FIGPROMPT", chr(92) + "ref{fig:prompt}") + "\n")
        if text is not None:
            out.append(verb(text))
    return "\n".join(out) + "\n"


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    # figures/fig_judge_panel.tex (Figure 2) is NOT written here while it is being redesigned as a heatmap
    # by scripts/iclr_fig_judge_heatmap.py (commit 26cc343); a later pass wires that script in and removes
    # judge_panel_figure(). Until then the committed file stands untouched (instruction of 2026-09-24).
    outputs = {"tab_routing.tex": routing_table(),
               "tab_form.tex": form_table(), "tab_form_main.tex": form_main_table(),
               "tab_baselines.tex": baselines_table(), "tab_human_anchor.tex": human_anchor_table(),
               "tab_framing_gen.tex": framing_gen_table(),
               "fig_framing_rubric.tex": framing_rubric_figure(),
               "fig_baseline_prompts.tex": baseline_prompts_figure(),
               "tab_rep4k.tex": rep4k_table(), "tab_pushback.tex": pushback_table(),
               "tab_seat.tex": seat_table(), "tab_runs.tex": runs_table(),
               "tab_run_stability.tex": run_stability_table(["claude-haiku-4-5", "gpt-4o"]),
               "tab_run_stability_b.tex": run_stability_table(["gpt-5.4-nano", "Llama-3.3-70B-Instruct",
                                                               "grok-4-1-fast-reasoning"]),
               "tab_z3.tex": z3_table(),
               "tab_interjudge.tex": interjudge_table(),
               "tab_knockout_readouts.tex": knockout_table(),
               "fig_collective_prompts.tex": collective_prompts_figure()}
    for name, text in outputs.items():
        (FIG / name).write_text(text)
        print("wrote", FIG / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
