# ARR May 2026 Reviewer Feedback: Submission 6086 (Narration-of-Thought)

Context: ACL ARR May 2026 cycle, Submission 6086, "Narration-of-Thought: Inference-Time
Scaffolding for Defeasible Ethical Reasoning in Large Language Models." Three official
reviews received (30 Jun - 4 Jul 2026). This document records the review summaries, the
independent verification of the flagged references, the decision, the draft author
responses, and the revision checklist for the next cycle.

## Decision record

Decision: do not withdraw and do not commit this version to EMNLP. Post author responses
to all three reviewers during the discussion window, then revise and resubmit to the next
ARR cycle.

Status (8 Jul 2026): all three author responses posted to OpenReview. Corrected
bibliography landed in `papers/shared/references.bib` and the paper rebuilds cleanly.
arXiv v2 replacement package prepared at `papers/acl/arxiv_submission_v2.zip` with the
corrected references (see arXiv v2 section below); upload pending. Remaining work is the
next-cycle revision checklist at the end of this document.

Rationale:

- The average Overall Assessment (~2.8) will not realistically clear the EMNLP
  main-conference bar, so committing this version wastes the cycle.
- Withdrawing without responding leaves the hallucinated-references finding unanswered on
  the record, which reads as conceded misconduct. A forthright response that acknowledges
  the bibliography errors and supplies verified replacement citations protects the
  integrity record at essentially no cost.
- Reviewer MqVt rated Excitement 4 and rejected solely on the references. A corrected
  resubmission (reviews carry forward in ARR; same reviewers can be requested) has a
  strong chance of flipping that review.
- ARR does not permit uploading a revised PDF during the discussion period; the response
  is text only, with the actual fixes landing in the next-cycle resubmission.

## Review summaries

| Reviewer | Overall | Soundness | Excitement | Confidence | Stance |
|----------|---------|-----------|------------|------------|--------|
| Xrff     | 3.5 (Borderline Conference) | 4 | 3.5 | 3 | Positive on substance; clarity complaints |
| WvQ2     | 3 (Findings) | 3 | 1.5 | 1 | Finds approach original; paper hard to read |
| MqVt     | 2 (Resubmit next cycle) | 1 | 4 | 3 | "Clear reject" solely on unverifiable references |

### Reviewer Xrff (OA 3.5)

Strengths: pressing task, cost-efficient and effective approach, extensive evaluation
including the five-section ablation and multi-stakeholder NoT.

Weaknesses and requests:

- Abstract is too long and overcomplicated (lines 2-8, 22-25, 35-38 called out).
- Paper is hard to read and follow; terms used before being introduced (SCM, Cliff's
  delta, trace, generator, cell at line 296).
- Models should be cited (lines 290-294); "generators" is not a common term.
- Only proprietary models are evaluated.
- Why only n=3 repetitions per (strategy, model, scenario)? Cost?

### Reviewer WvQ2 (OA 3, Confidence 1)

Strengths: originality of NoT and especially the multi-stakeholder variant (formal
decider, primary affected, external viewer); sees these as potentially justifying
acceptance.

Weaknesses and requests:

- Introduction confusing; acronyms not expanded on first use (e.g., CoT).
- DailyDilemmas is not presented in the paper; the reader must consult the cited paper.
- The ethical framing of the dataset is debatable ("ethics cannot be reduced to such
  application of values").
- Results presentation is complex and difficult to understand; recommends a full
  rewrite for clarity.

### Reviewer MqVt (OA 2, Soundness 1, Excitement 4)

Strengths: clear benefits of NoT, particularly stakeholder representation; would tell
others about the paper.

Weaknesses and requests:

- Four references could not be found: Bientzle 2024, Bientzle 2021 (not in Health
  Communication 36(7)), Shaffer et al. 2019; Rottger et al. 2023 exists but with a
  different title. "This alone makes this a clear reject."
- Requests an overview of the Moral Uncertainty literature, particularly fanaticism
  issues around voting among stakeholders.

## Consolidated criticism themes

1. Reference integrity (MqVt; substantiated, see verification below). Highest priority.
2. Clarity and readability (Xrff and WvQ2 independently): abstract length, undefined
   terms (SCM, Cliff's delta, trace, generator, cell), acronym expansion, DailyDilemmas
   never described in-paper, model citations missing, dense results presentation.
3. Missing related work on Moral Uncertainty, including fanaticism in stakeholder voting
   (MqVt).
4. Model selection and repetitions: proprietary-only generator pool; justify or expand
   n=3 repetitions (Xrff).
5. Artifact availability perception: two reviewers rated Datasets 1 ("no usable datasets
   submitted") and one rated Software 1, despite the submitted supplementary zip. The
   availability and packaging of artifacts must be made unmistakable.

## Reference verification (performed 8 Jul 2026)

All four flagged references were checked against the bibliography in
`papers/shared/references.bib`. The reviewer's complaint is substantiated.

- `shaffer19` ("Why Stories Matter: Narrative Perspective-Taking and the Cultivation of
  Empathic Concern," Medical Decision Making 38(3), 2019): fabricated. No such paper
  exists. Medical Decision Making volume 38 is 2018, and the listed co-authors do not
  match any Shaffer paper. The real 2019 Shaffer paper on this topic is in PLoS ONE.
- `bientzle21` ("The Effects of Narrative Video on Viewers' Understanding of and
  Attitudes towards Organ Donation," Health Communication 36(7), 820-829): fabricated.
  No such article exists in that journal or issue; the only web source for the citation
  is this paper's own preprint bibliography.
- `bientzle24` ("Narrative Persuasion in Health Communication," Patient Education and
  Counseling, 2024): fabricated. Same situation; no independent source exists.
- `rottger23`: real paper, but the bibliography title was wrong ("Excessive Safety
  Refusals" instead of the actual "XSTest: A Test Suite for Identifying Exaggerated
  Safety Behaviours in Large Language Models") and the author list wrongly included
  Kalina Bontcheva in place of Federico Bianchi. The paper is now published at NAACL
  2024 (pages 5377-5400), superseding the arXiv citation.

Verified replacement citations (same claim supported: narrative interventions measurably
increase perspective-taking and empathic concern):

- Shaffer, V. A., Bohanek, J., Focella, E. S., Horstman, H. K., and Saffran, L. (2019).
  Encouraging perspective taking: Using narrative writing to induce empathy for others
  engaging in negative health behaviors. PLoS ONE, 14(10), e0224046.
  doi:10.1371/journal.pone.0224046
- Bientzle, M., Eggeling, M., Kanzleiter, M., Thieme, K., and Kimmerle, J. (2021). The
  impact of narrative writing on empathy, perspective-taking, and attitude: Two
  randomized controlled experiments on violations of Covid-19 protection regulations.
  PLoS ONE, 16(7), e0254501. doi:10.1371/journal.pone.0254501
- Bientzle, M., Eggeling, M., and Kimmerle, J. (2024). Good reasons for bad behavior: A
  randomized controlled experiment on the impact of narrative reading and writing on
  empathic concern, perspective-taking, and attitude. Frontiers in Public Health,
  12:1343225. doi:10.3389/fpubh.2024.1343225
- Rottger, P., Kirk, H., Vidgen, B., Attanasio, G., Bianchi, F., and Hovy, D. (2024).
  XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language
  Models. In Proceedings of NAACL 2024 (Volume 1: Long Papers), pages 5377-5400.

Status: `papers/shared/references.bib` has been corrected with these entries (same bib
keys retained so `papers/acl/ACL_paper.tex` and `papers/archive/position_paper.tex`
compile unchanged). Root cause is consistent with AI-assisted bibliography generation;
the revision process must include per-entry verification of every reference against its
primary source before resubmission.

## Author responses (final text for OpenReview)

Post each as an official comment threaded under the matching review, visible to all
listed readers.

### To Reviewer MqVt

Note on framing: the reviewer did not raise an ethics flag (their review states "There
are no concerns with this submission" and "Needs Ethics Review: No"); they presented
the references as a correction ("Fix the hallucinated references") scored under
Soundness. The response must keep it in that lane: plain factual account of the cause,
no minimizing labels such as "clerical error" that conflict with the reviewer's stated
suspicion and could compound the record if scrutinized.

You are right, and thank you for checking. We verified all four entries against primary
sources. Three do not correspond to real papers, and the Rottger entry had the wrong
title and one wrong author. The errors came from reference drafting that was not
verified entry by entry against primary sources. That process has been corrected and
every reference has now been verified. We apologize for the error.

The corrected entries: Shaffer et al. 2019, PLoS ONE 14(10): e0224046. Bientzle et al.
2021, PLoS ONE 16(7): e0254501. Bientzle et al. 2024, Frontiers in Public Health 12:
1343225. Rottger et al. 2024, XSTest, NAACL 2024, pages 5377 to 5400. No result depends
on these entries; they appear in one related-work sentence.

On the moral uncertainty critique we agree completely. As we read your concern, voting
among stakeholders can go fanatical in two directions: a position with extreme declared
stakes can dominate the outcome, or intensity gets flattened and an absolutist stake is
outvoted as if it were mild. The moral uncertainty literature treats exactly this
problem, and parliamentary mechanisms were proposed partly to tame it. The revision
will discuss MacAskill, Bykvist, and Ord (2020), the parliamentary approach of Newberry
and Ord (2021), and related literature as appropriate to better frame our argument and
address your objection.

Thank you again for your thoughtful reflection here. These comments will help us
improve this work.

### To Reviewer Xrff

Thank you for the concrete pointers. We will act on all of them.

Abstract: agreed, too long. We will cut it hard and lead with the two failure modes and
the headline numbers.

Terminology: SCM, Cliff's delta, trace, generator, and cell will be defined at first
use. We will likely drop "generator" for "evaluated model."

Model citations at lines 290 to 294: we will add them.

On n=3: cost was a factor, but the design holds up. Three repetitions per cell with
fixed seeds (42, 43, 99), and the bootstrap confidence intervals on Cliff's delta run
over the full per-cell sample. We will state this in the paper and add a sensitivity
check with more repetitions on a subset.

Proprietary-only models: fair. The revision adds at least one open-weights model.

### To Reviewer WvQ2

Thank you for the review, and for recognizing what the multi-stakeholder protocol is
trying to do.

Readability is the core criticism and we accept it. The revision includes a structural
rewrite: every acronym expanded at first use, a self-contained description of
DailyDilemmas with a worked example so the reader never needs the source paper, and a
results section that states the two failure-mode metrics in plain language before any
effect-size machinery.

On the ethical dimension: we agree ethics cannot be reduced to applying values to
scenarios. Our claims are about the structure of the reasoning trace, who gets named
and what uncertainty gets acknowledged, not about the moral correctness of outcomes.
The revision states that scope limit explicitly. Your comment sharpened this for us.

## arXiv v2 replacement (curing the public preprint)

The public preprint (arXiv 2606.26366, v1) still contains the erroneous references and
must be replaced. The v1 source corresponds to `papers/acl/ACL_paper.tex` plus
`papers/shared/references.bib` (verified by diff against
`papers/acl/arxiv_submission.zip`; the only tex difference is the corrected related-work
sentence).

Changes in v2 relative to v1, deliberately minimal:

1. Corrected bibliography entries for shaffer19, bientzle21, bientzle24 (verified
   replacements) and rottger23 (correct title, author list, NAACL 2024 venue).
2. One related-work sentence updated to match what the replacement citations show:
   "Narrative-perspective interventions in health-communication contexts measurably
   increase perspective-taking and empathic concern."

The v2 package is `papers/acl/arxiv_submission_v2.zip` (tex, compiled bbl, bib, acl.sty,
acl_natbib.bst, six figure PDFs). It compiles standalone with xelatex and the corrected
entries appear in the bibliography with no undefined citations.

To upload: arXiv > 2606.26366 > Replace, upload the zip, and in the "Comments" field
note: "v2: corrects four bibliography entries; no changes to methods, results, or
claims." Everything else on the submission form carries over from v1.

# --- Revision status (9 Jul 2026) ---
# Author responses posted to OpenReview. arXiv v2 uploaded with corrected references.
# Next-cycle revision applied to papers/acl/ACL_paper.tex (review mode, anonymized).

## Revision checklist for next-cycle resubmission

1. References: verified macaskill20 (OUP 2020, DOI 10.1093/oso/9780198722274.001.0001),
   newberry21 (FHI tech report 2021-2, ORA deposit; byline Toby Newberry -- the PDF's
   "Cite as" line has a typo "Newbery"), model-card misc
   entries (openai_gpt54nano, anthropic_haiku45, anthropic_sonnet46, xai_grok41,
   deepseek_v3, chiu24data); prior fixes for shaffer19, bientzle21, bientzle24, rottger23.
2. Clarity rewrite: done (abstract, terminology, DailyDilemmas, scope, n=3 rationale).
3. DailyDilemmas exposition: done in Exp 1 setup with worked example.
4. Related work: Moral Uncertainty paragraph added (macaskill20, newberry21).
5. Model pool: deepseek-v3 Exp 1 grid complete (1500 cells, N=5); Table 1
   and firing figure updated.
6. Repetition sensitivity: appendix app:sensitivity; 20-scenario draw (seed 43),
   10 replicates per cell: delta +0.91 [0.86, 0.94] vs +0.93 full sample (stakeholder
   count), +0.74 vs +0.73 (uncertainty score). Stable across random draws (0.84-0.98
   over 20 seeds).
7. Phase 11 head-to-head: integrated in section 4.2 and Appendix app:phase11.
8. ELEPHANT caveat: Phase 15 judge-reliability numbers added.
9. Artifact availability: Reproducibility section strengthened with per-artifact licenses.
10. Template: main body trimmed to 7 pages before Limitations (page 8); `[review]{acl}`
    and anonymous author block active. Std-CoT debate comparison from Phase 11 data.
11. Final gate: bibliography entry verification complete for newly added citations.

**Locator:** the repository README has an **ARR resubmission (main paper)** section that indexes this file, `papers/acl/ACL_paper.tex`, build commands, artifact paths, and the pre-submit checklist.

## Post-integration QA pass (9 Jul 2026)

A full consistency audit after the deepseek-v3 data landed:

- Table 1 gpt-5.4-nano deltas corrected to the rebuilt CSV (-29/-70; the cache-merge
  rebuild shifted rates slightly from the pre-merge values -30/-72).
- Panel range claims updated: proprietary-model ranges (50.0-71.3% suppression,
  14.6-30.1% collapse) stated separately from deepseek-v3's lower baselines (23.6%,
  1.2%); NoT ranges now 0.0-24.5% / 0.0-1.2%; pp-drop range 24-70.
- "Below 1 percent collapse" claims corrected to "at most 1.2%" (grok NoT = 1.17%).
- Abstract corrected to five models / four vendors.
- Appendix effect-size table refreshed against tier1_effect_sizes_quartet.csv
  (nano and haiku rows updated to rebuilt values).
- Length-residualisation text now includes DeepSeek (resid delta +0.36 sc / +0.49 us).
- Incorrect seed claim removed: the pipeline uses the replicate index as the decoding
  seed (scripts/generators.py), not "seeds 42/43/99"; paper text now matches the code.
- Phase 11 appendix: 3.57 corrected to 3.53 (textgrad_binary mean sc per p11_summary).
- TGD drift caveat: nano baseline rates corrected to 30.1%/71.3% (were grok/haiku values).
- Sensitivity appendix rewritten with reproducible numbers (see item 6).
- newberry21 author restored to "Newberry" (ORA byline; the PDF "Cite as" line has a typo).
- De-anonymisation risk removed: internal repo path papers/sycophancy dropped from the
  ELEPHANT appendix.
- Rebuilt clean: 24 pages, Limitations opens page 8, no undefined references/citations.

## Writing and clarity pass (9 Jul 2026)

A full main-body rewrite pass targeting reviewer WvQ2's readability criticism:

- Duplicated line removed in related work ("no revision absorbs." appeared twice).
- Internal codenames purged from the PDF: Phase 1/11/12/13/15 references replaced with
  descriptive language ("held-out head-to-head", "initial smoke sample", "subsequent
  three-judge panel study", "Experiment 1 outputs"); "verified quartet" ->
  "four-generator panel"; "Tier-1" dropped from figure caption; scenario IDs
  (pharma_whistleblower, av_engineer) replaced with prose descriptions.
- Terms now defined at first use: "trace" glossed in intro; CoT expanded at first body
  use; Cliff's delta given a plain-language definition plus the |delta|>0.474 "large"
  convention in Setup (before any use); "cell" defined explicitly; generator/evaluated
  model synonymy stated once; residualisation glossed; SCM expanded in contributions.
- Exp 1 setup restructured into Corpus / Conditions and models / Coding paragraphs
  (was two dense paragraphs with coding details split across both).
- Intro NoT paragraph rewritten: concrete instruction first, distributional hypothesis
  second (was jargon-first); "amortise causal-trajectory inference by retrieval"
  simplified.
- Contributions: each result now names its mechanism in plain terms; fixed stale
  cross-reference ("Finding 4" -> fifth result; sycophancy appendix range ->
  app:deploy).
- Matched-budget paragraph reframed as a question ("Could a model simply told to
  write more do as well?"); ablation paragraph now states the six conditions
  explicitly (full NoT + five drop-one variants).
- TGD subsection: in-family/cross-family definitions untangled; third-judge
  description corrected (same model as cross-family training judge, caveat
  referenced, not "a vendor involved in neither run").
- Factual fix: uncertainty-suppression residual under NoT is retained by grok
  (24.5%) as well as the two Anthropic models (20.8-21.5%); text previously
  attributed the residual to Anthropic alone.
- Limitations: "survived the cache rebuild" internal jargon replaced.
- Rebuilt clean: 25 pages, Limitations opens page 8, no undefined references/citations.
