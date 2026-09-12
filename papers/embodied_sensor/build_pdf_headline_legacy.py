"""RETIRED. The headline-PDF builder from before the ICLR figures (make_figures.py,
2026-09-12). Kept for the record of Embodied_Deliberation_Error_Sensor.pdf (built 09:03
the same day, before the figure regeneration); NOT runnable as it stands and not to be
rerun without rewriting:

  * img() reads the PNGs from a session scratchpad path that no longer exists, not from
    papers/embodied_sensor/figures/;
  * it embeds fig2.png, which make_figures.py does not produce (results.tex header
    note (d): fig2 is not referenced and not regenerated from any cited source);
  * its Figure 4 caption describes panels A/B/C with a pairwise-phi panel and its
    Figure 5 caption describes "the model's own sampling disagreement" points; the
    regenerated fig4.png has panels A (dissent concentration) and B (one effective
    voter), and fig5.png has the haiku (A) and sonnet (B) judge panels only;
  * its R5 prose quotes the composite stake-aware flag's pre-artefact intervals
    (+0.201 [+0.075, +0.346] and the like); 16.15.3 RESULTS in the prereg says the
    paper must quote the artefact's bounds (transfer_readouts.json), never the R5
    quotes, and the registered sensor of 16.15 is the stake-blind counter.

The ICLR submission's figures are built by make_figures.py and captioned in results.tex;
this file predates both. Renamed from build_pdf.py on 2026-09-12 (zero spend).
"""
import base64, subprocess, os
S = "/private/tmp/claude-501/-Users-pat-code-ANI-Examination/038f1242-0ad8-4465-ab0d-350428a1abc8/scratchpad"
def img(n):
    return "data:image/png;base64," + base64.b64encode(open(f"{S}/{n}", "rb").read()).decode()

HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Embodied Deliberation as an Error Sensor</title>
<style>
@page {{ size: A4; margin: 18mm 17mm 18mm 17mm; }}
body {{ font-family: -apple-system, "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 10.4pt; line-height: 1.42; color: #111; max-width: 100%; }}
h1 {{ font-size: 19pt; line-height: 1.2; margin: 0 0 4pt 0; }}
.sub {{ color: #444; font-size: 10pt; margin-bottom: 14pt; }}
h2 {{ font-size: 13.5pt; margin: 18pt 0 6pt 0; border-bottom: 1.5px solid #1f4e79; padding-bottom: 2pt; color: #1f4e79; }}
h3 {{ font-size: 11pt; margin: 12pt 0 4pt 0; }}
p {{ margin: 0 0 7pt 0; }}
.box {{ border-left: 4px solid #1f4e79; background: #f3f6fa; padding: 8pt 11pt; margin: 8pt 0 12pt 0; }}
.box.warn {{ border-left-color: #c0392b; background: #fbf2f0; }}
.claim {{ margin: 0 0 8pt 0; padding-left: 26pt; text-indent: -26pt; }}
.claim b.n {{ display: inline-block; width: 22pt; text-indent: 0; color: #1f4e79; }}
table {{ border-collapse: collapse; width: 100%; font-size: 9.3pt; margin: 6pt 0 10pt 0; }}
th, td {{ border-bottom: 1px solid #ccc; padding: 3pt 6pt; text-align: left; vertical-align: top; }}
th {{ background: #eef2f7; font-weight: 600; }}
td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
figure {{ margin: 10pt 0 12pt 0; page-break-inside: avoid; }}
figure img {{ width: 100%; }}
figcaption {{ font-size: 9pt; color: #333; margin-top: 3pt; }}
figcaption b {{ color: #1f4e79; }}
.small {{ font-size: 8.8pt; color: #444; }}
.pb {{ page-break-before: always; }}
code {{ font-family: Menlo, Consolas, monospace; font-size: 8.8pt; background: #f2f2f2; padding: 0 3pt; }}
ul {{ margin: 2pt 0 8pt 18pt; padding: 0; }} li {{ margin-bottom: 3pt; }}
.two {{ display: flex; gap: 14pt; }} .two > div {{ flex: 1; }}
</style></head><body>

<h1>Embodied Deliberation as an Error Sensor</h1>
<div class="sub">Role-lock, the effective voter count, and a graph theory of where collectives locate their own mistakes<br>
Working results, 2026-09-12 · branch <code>sycophancy-scenario-redesign</code> · every number recomputed in-session from cached artefacts; provenance in §7</div>

<div class="box">
<b>Headline.</b> Assigning opposed first-person interests to LLM agents is a near-total control on what each agent concludes, and that control has a price and a product. The price: a role-locked agent's verdict carries almost no information about the case, so a collective built from such agents has an effective voter count of one and loses to a single self-consistency call. The product: the same role structure organises the agents' disagreement into a signal that locates hard items, and that signal <i>transfers</i> — it concentrates the errors of independent generators it never touched, including one seven points stronger than the deliberation itself. The collective is a poor generator and a good detector, and one mechanism does both.
</div>

<h2>1. Headline results</h2>

<p class="claim"><b class="n">R1</b><b>Embodiment is a near-total directional control.</b> Role-lock — |P(at-fault) for the writer's advocate − P(at-fault) for the counterparty| at the opening statement — is <b>0.971</b> on 1,666 grok debates: the advocates return their assigned side on 99.7% and 97.4% of statements. Lock is graded across models (sonnet 0.677, nano 0.620, haiku 0.442). Removing the <i>stake</i> clause while keeping the <i>directional</i> assignment leaves lock at 0.946 (n=377): the pinning force is being told whose case to make, not having something to lose.</p>

<p class="claim"><b class="n">R2</b><b>The control destroys the agent's verdict-information.</b> Locked seats sit at the base rate — writer's advocate 0.592, counterparty 0.437 (below chance) — while the unlocked neutral seat scores 0.857. Across four models, a seat's verdict accuracy rises as its lock falls (Fig. 1B).</p>

<p class="claim"><b class="n">R3</b><b>The collective's effective voter count is one.</b> The three-seat majority equals the neutral seat's verdict on <b>1,663 of 1,663</b> debates. A cross-validated oracle over everything the panel emits (nine seat verdicts, both moderator verdicts, all labels and votes) scores 0.831, <i>below</i> the neutral seat alone at 0.852. The deployed moderator synthesis (0.844) is worse than the neutral seat's own opening statement (0.857; +0.014 [+0.000, +0.028]). One grok call at majority-of-3 (0.912) beats the 17-call deliberation (0.843) by <b>+0.069 [+0.043, +0.097]</b>; scaffold-matched, +0.053 [+0.029, +0.079].</p>

<p class="claim"><b class="n">R4</b><b>The dissent is structured and diagnostic.</b> Objections are anti-correlated on the advocate pair (φ = −0.72): exactly one seat objects on 79% of debates against 50% under the correct heterogeneous-marginal independence null (localisation excess <b>+0.291</b>). Where two or more seats object, the deliberation is wrong 42% of the time versus 12% otherwise (<b>3.5×</b>); the counter needs no stake bookkeeping to read (it is a strict subset of the stake-aware flag, 328/337). Dissent informativeness tracks grip across models — lift +0.320 / +0.133 / +0.039 for G3 +0.631 / +0.062 / +0.034 — and not accuracy (Fig. 2).</p>

<p class="claim"><b class="n">R5</b><b>The signal transfers to generators it never touched.</b> The same flag concentrates the errors of grok answering alone (accuracy 0.912): P(wrong | flagged) 0.264 vs 0.063, <b>4.2×</b>, lift +0.201 [+0.075, +0.346]; grok narrative solo 4.8× [+0.132, +0.413]; nano 2.3× [+0.060, +0.360]; haiku 1.8× [−0.018, +0.256]. The flag locates hard <i>items</i>, not the deliberation's own idiosyncrasies (Fig. 3).</p>

<p class="claim"><b class="n">R6</b><b>Acting on the signal pays, and the two sensors combine.</b> Routing flagged debates to an independent cross-vendor judge raises the flagged subset by +0.193 [+0.067, +0.310] and the composed system to 0.881 — 3.75 points above random routing at matched cost. Embodied dissent and the model's own sampling disagreement find <i>different</i> hard items (274 vs 69 exclusive firings, 63 shared); their union routes at 24% coverage to 0.888, beating route-everything by +0.038 [+0.006, +0.072], which the embodied flag alone could not certify (Fig. 5).</p>

<p class="claim"><b class="n">R7</b><b>The collective cannot be steered to truth from inside.</b> Feeding the flag back and re-running the same agents nets +0.012 [−0.039, +0.063]; every internal corrective rung is null (statement-against-interest exactly 0.000); a majority over three passes does not beat one. Objectivity, where obtained, came from an external decorrelated model overriding the collective where it flagged itself.</p>

<figure><img src="{img('fig1.png')}">
<figcaption><b>Figure 1. Embodiment as control, and its cost.</b> (A) Probability that each seat's opening verdict is at-fault, four models. Grok's advocates are constants of their role; the neutral seat sits at the gold base rate. (B) A seat's verdict accuracy against its role-lock, all four models: the harder the seat is pinned, the less its verdict knows about the item. The writer's advocate goes from 0.816 (haiku, lock 0.44) to 0.592 (grok, lock 0.97).</figcaption></figure>

<figure><img src="{img('fig2.png')}">
<figcaption><b>Figure 2. The transfer function.</b> Error lift of the stake-blind "2+ seats objected" counter against stake concentration (grip, G3), one point per model, marker size ∝ accuracy. The ordering is grip's (grok &gt; haiku &gt; nano), not accuracy's (haiku &gt; grok &gt; nano). Nano's error rate is within 1.4 points of grok's; its dissent simply does not land on its errors. Sonnet omitted (31 items, 31/31 correct, unevaluable).</figcaption></figure>

<figure><img src="{img('fig3.png')}">
<figcaption><b>Figure 3. The headline: embodied dissent transfers.</b> (A) Error rate among debates the embodied flag fires on versus not, for the deliberation itself (reference) and four generators the flag was never computed from. (B) The corresponding lifts with 95% item-clustered bootstrap CIs. Three of four transfer targets exclude zero; the weakest (haiku) is cross-vendor. The flag is a property of the <i>item</i>, revealed by role-structured disagreement, not a property of the deliberation's own errors.</figcaption></figure>

<figure><img src="{img('fig4.png')}">
<figcaption><b>Figure 4. The effective voter count is one.</b> (A) Distribution of the number of objecting seats, observed versus the Poisson-binomial independence null built from the per-seat marginals (0.430 / 0.690 / 0.073). Dissent concentrates in exactly one seat far more than chance. (B) Pairwise φ on objections: the anti-correlation is one edge — the two opposed advocates — and is mechanically forced (a committal verdict undermines exactly one of two opposed stakes); the neutral seat is nearly independent of both. (C) No function of the panel beats its stakeless member; the deployed synthesis is below it; one solo call is above all of them.</figcaption></figure>

<figure><img src="{img('fig5.png')}">
<figcaption><b>Figure 5. Sensor, not generator.</b> Composed-system accuracy against the fraction of debates routed to an external judge, all routers using the same judge. Selection matters (random routing at matched coverage loses 3.75 points); the embodied flag and the model's own sampling disagreement find different items and combine; but the whole composed apparatus sits below one grok call. The sensor is right about where the base is wrong (Fig. 3); the judge is too weak to act on it. The actuator, not the sensor, is the open engineering problem.</figcaption></figure>

<h2 class="pb">2. The theory of graph structure — the substantiated core</h2>

<p>A theory of "graph structures for communities of embodied agents reaching objective consensus" needs three things: an order parameter it predicts, the structural variables it predicts from, and a formal reason the relation should hold. All three are now in hand for one topology, and the first result of the theory is negative in the direction the programme initially hoped and positive in a direction it did not anticipate.</p>

<h3>2.1 A node serves two separable functions</h3>
<p>Every node in a deliberation graph can contribute in two ways that the literature has not kept apart: by being <i>right</i> (verdict-information, what its answer says about the item) and by <i>revealing that something is wrong</i> (dissent-information, what its disagreement says about whether the group has erred). R2 and R4 show that on this topology these live in different nodes. All of the verdict-information is in the one stakeless node; all of the dissent-information is in the two locked ones.</p>

<h3>2.2 Role assignment is a transfer function between them</h3>
<p>Assigning a node a directional interest converts its verdict-information into dissent-information, and the conversion is graded by how hard the role pins the node. Role-lock is the control variable. At lock 0.97 (grok) the advocates' verdicts are constants of their role and carry ~0 bits about the item, while their objections carry a +0.320 error lift; at lock 0.44 (haiku) the advocate's verdict is informative (0.816) and its objections carry +0.133. This is the relation in Figs. 1B and 2, measured across four models on one instrument, and it explains why the best sensor model is the worst collective.</p>

<h3>2.3 The dissent structure is an edge property, and the edge is mechanical</h3>
<p>The anti-correlation that makes dissent rare and diagnostic lives on one edge — the pair of opposed advocates (φ = −0.72) — not on the graph. It is forced: any committal verdict undermines exactly one of two opposed stakes, so the two seats cannot both be undermined and cannot both be satisfied. The neutral node is nearly independent of both (−0.14, +0.15). A theory of this graph is therefore a theory of that one edge plus a hub, which is why the 16.8 ablation (stake removed, direction kept) left the structure intact: the edge is created by <i>opposed direction</i>, and 16.10 tests whether it needs the exchange at all.</p>

<h3>2.4 Why this graph cannot be a wise generator: Golub–Jackson</h3>
<p>Golub and Jackson (2010, <i>AEJ: Microeconomics</i>) prove that a network's consensus converges to the truth <i>if and only if</i> the influence of the most influential node vanishes. The condition is violated here by construction: the neutral node is a literal dictator (R3, 1,663/1,663), because the two locked nodes cancel and leave it the tiebreaker on every debate. A dictator's "consensus" is that node's belief, not a crowd's; the negative oracle gap (−0.021) is the empirical signature. This is also why iteration cannot converge (R7): re-running the same three agents re-samples the dictator's posterior. Any successor topology that wants the collective to be a generator must satisfy the no-dominant-node condition, and the current one fails it <i>because</i> of the same role assignment that makes it a good sensor.</p>

<h3>2.5 The order parameter</h3>
<p>The quantity a graph theory of objectivity should predict is <b>Λ, the effective number of conditionally independent competent voters in the verdict layer</b>, operationalised as the oracle gap (best achievable aggregation minus best single node; measured −0.021) together with role-lock (which sets how many nodes carry verdict-information at all). Λ = 1 on this topology. The localisation excess of objections (+0.291) is the complementary order parameter for the dissent layer. A design that raises Λ above 1 while preserving localisation is the target, and the theory says the two pull against each other through role-lock.</p>

<div class="box">
<b>The substantiated statement, in one paragraph.</b> On a three-seat, complete-graph, externally-moderated deliberation, directional role assignment pins two nodes to their roles (lock 0.97), which destroys their verdict-information, collapses the effective voter count to one, and — by Golub–Jackson — makes the collective structurally incapable of converging to truth. The same assignment creates an anti-correlated dissent edge between the two locked nodes, whose firing locates hard items with 3.5–4.8× error concentration, transfers to independent generators, and is complementary to the model's own sampling disagreement. The graph's value is in its edge, not its vertex aggregate; it is a sensor to be placed over a separate generator, not a generator.
</div>

<h2>3. What is <i>not</i> claimed, and was withdrawn on the evidence</h2>
<ul>
<li><b>Not</b> "the collective converges toward objective conclusions." It cannot (§2.4, R7); its own registered check reads "NOT CONFIRMED, and probably the wrong place to look."</li>
<li><b>Not</b> "the stake structure is what selects the errors at readout." Counting objections does that; stake bookkeeping adds 9 of 337 routed debates.</li>
<li><b>Not</b> "dissent marks where a second opinion will help rather than hurt." It concentrates errors 4.6×; the judge repairs only slightly better there (0.645 vs 0.593) and damages correct answers slightly more (0.133 vs 0.104).</li>
<li><b>Not</b> "embodied dissent is a sharper sensor than self-consistency." Per routed case, sampling disagreement is sharper (precision 0.561 vs 0.418). The claim is <i>complementarity</i>, not superiority.</li>
<li><b>Not</b> "the composed system beats a strong solo." It does not (0.888 vs 0.912). The sensor is correct about where the solo is wrong; the available actuator is too weak to fix it.</li>
<li><b>Not</b> "grip is a property of LLMs." It is grok-specific among four models; teachable to others by few-shot distillation, but taught dissent has not driven an actuator.</li>
</ul>

<h2>4. Relation to the proximal literature (primary sources verified)</h2>
<table>
<tr><th style="width:26%">Work</th><th style="width:37%">What it established</th><th>Where this work sits relative to it</th></tr>
<tr><td>Mercier &amp; Sperber 2011, <i>BBS</i> 34(2); 2017 <i>The Enigma of Reason</i></td><td>Reasoning evolved for argumentation; individual reasoning is biased advocacy (confirmation bias is a feature of an arguer); truth emerges from the <i>evaluation</i> side of the exchange.</td><td>Predicts R1–R2 exactly (advocates are locked, uninformative producers) and predicts our architecture (production biased, evaluation elsewhere). Does not predict that advocates converge — and ours do not. Closest conceptual precedent.</td></tr>
<tr><td>Golub &amp; Jackson 2010, <i>AEJ: Micro</i> 2(1)</td><td>Network consensus converges to truth iff no node's influence dominates.</td><td>Formal derivation of R3 and R7. Names the condition the current topology violates and any successor must satisfy. Closest formal precedent.</td></tr>
<tr><td>Smit et al. 2024, ICML, PMLR 235</td><td>Multi-agent debate does not reliably outperform self-consistency or ensembling on accuracy.</td><td>R3 replicates this on a new instrument and supplies the mechanism (role-lock → Λ = 1). But they measured the wrong dependent variable for our claim: R4–R6 show debate's value is in <i>error detection</i>, which they did not test. We cite them as proximal and move past them there.</td></tr>
<tr><td>Li et al. 2024, Findings EMNLP</td><td>Sparse communication topologies match or beat fully-connected debate at lower cost.</td><td>The nearest empirical treatment of topology as a variable; accuracy as outcome, no order parameter. Our 2×2 (§5) varies connectivity with the dissent structure, not accuracy, as the readout.</td></tr>
<tr><td>Irving, Christiano &amp; Amodei 2018</td><td>AI safety via debate: two opposed agents, a judge decides.</td><td>The judge, not the debaters, is the locus of truth in their framing — closer to R7 than it first appears. Our contribution is measuring what the debaters carry (nothing) and what their disagreement carries (a transferable item-difficulty signal).</td></tr>
</table>
<p class="small">A five-strand sweep with independent primary-source re-verification of every citation is in progress (aggregation theory incl. Ladha, DeGroot, Hong–Page, Prelec; persona/activation steering; social epistemology incl. Longino; the topology literature incl. MacNet, GPTSwarm). This programme's prior submission was rejected for fabricated references; no citation enters the manuscript without a fetched primary source.</p>

<h2>5. The programme — labelled as such — and what is running now</h2>
<p><b>Independent variables:</b> role assignment (directional / stake / identical) and connectivity (who reads whom), then seat count, moderator position, aggregation rule. <b>Order parameters:</b> Λ (oracle gap, role-lock) in the verdict layer; localisation excess in the dissent layer; transfer lift as the deployment-relevant readout. <b>Design principle that follows:</b> separation of functions — an embodied graph as sensor over a distinct generator that satisfies the no-dominant-node condition, or is a strong solo.</p>
<p><b>Launched 2026-09-12 (Addendum 16.10, $24 ceiling), the role × connectivity 2×2 on the same 210 items:</b></p>
<table>
<tr><th></th><th>edges ON</th><th>edges OFF (no seat reads any other)</th></tr>
<tr><td><b>embodied roles</b></td><td>exists (1,680 debates)</td><td><i>running</i> — do the peer edges do anything?</td></tr>
<tr><td><b>identical roles</b> (neutral ×3)</td><td><i>running</i> — does direction produce the structure?</td><td><i>running</i> — the floor: no roles, no graph</td></tr>
</table>
<p>The 2×2 separates H-ROLE (structure comes from role assignment, survives edge removal) from H-EDGE (structure comes from the exchange, survives role removal). Every cell reports role-lock, localisation excess, fire rate, G3 where stakes exist, the oracle gap, and transfer to grok-solo. With identical seats a majority is a genuine aggregation for the first time, so Λ can exceed 1. Pre-declared readings, gates and MDEs are in the registration.</p>
<p><b>Next after the 2×2:</b> a stronger actuator (the sensor finds where a 0.912 base is wrong; a 0.850 judge cannot fix it — a stronger model or a human closes the loop); a differentiated-non-directional role set if the identical cell collapses; seat count and moderator-inside as the next structural variables.</p>

<h2>6. Corrections made on the record this cycle</h2>
<ul class="small">
<li>Localisation excess restated from +0.359 to <b>+0.291</b>: the earlier figure used a pooled per-seat objection rate; the seats' marginals are 0.430 / 0.690 / 0.073 and the Poisson-binomial null is the correct one.</li>
<li>The anti-correlation is one edge (advocate pair), not a graph-wide property.</li>
<li>Addendum 14 ($48 Dilemmas growth) cancelled: at its own target the half-width was 4.5× the effect, and the corpus cannot supply a powered n at any price.</li>
<li>The 16.8 readout module was corrected before any number was reported: AITA's five-token instrument must be scored by the registered <code>code_response</code> collapse, not string equality (0.7173 vs 0.8427); and the baseline restricted to the control's surviving items.</li>
</ul>

<h2>7. Provenance</h2>
<p class="small">R1, Fig. 1: r0 cache replay, <code>cgd_*_r0_*.json</code>, four panels. R2–R3: <code>analyze_actuator_ladder.load_debates</code> + <code>replay_debate</code>; solo comparators from <code>cg_scaffold_combined_rows_k3.csv</code>. R4, Fig. 4: <code>cg_deliberation_{{rows,votes}}.csv</code>; <code>claim_audit_analysis.json</code> (Addendum 16.1–16.2). R5, Fig. 3 and R6, Fig. 5: composed-system and transfer computations against the same cached judge, this session. R7: prereg L2 RESULTS, <code>actuator_ladder_analysis.json</code>. 16.8 ablation: <code>unembodied_ablation_analysis.json</code>. All CIs are 95% item-clustered bootstrap, 3,000–4,000 draws. Registration and results: <code>Guidance_Documents/prereg_embodiment_community.md</code>, Addenda 16–16.10.</p>

</body></html>"""

open(f"{S}/headline_theory.html", "w").write(HTML)
chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
out = f"{S}/Embodied_Deliberation_Error_Sensor.pdf"
r = subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={out}", f"file://{S}/headline_theory.html"],
                   capture_output=True, text=True, timeout=120)
print("chrome rc", r.returncode, os.path.exists(out) and os.path.getsize(out))
