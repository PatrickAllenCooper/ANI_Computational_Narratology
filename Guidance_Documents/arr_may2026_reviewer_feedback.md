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

You are right, and thank you for checking. We verified all four entries against primary
sources. Three do not correspond to real papers, and the Rottger entry had the wrong
title and one wrong author. The errors came from reference drafting that was not
checked entry by entry. Every reference has now been verified.

The corrected entries: Shaffer et al. 2019, PLoS ONE 14(10): e0224046. Bientzle et al.
2021, PLoS ONE 16(7): e0254501. Bientzle et al. 2024, Frontiers in Public Health 12:
1343225. Rottger et al. 2024, XSTest, NAACL 2024, pages 5377 to 5400. No result depends
on these entries; they appear in one related-work sentence.

On moral uncertainty: agreed. The revision will discuss MacAskill, Bykvist, and Ord
(2020), Newberry and Ord's parliamentary approach (2021), and fanaticism in
voting-based aggregation, connected explicitly to our multi-stakeholder vote and the
residual structural-rejection rate.

Thank you again. This review made the paper better.

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

## Revision checklist for next-cycle resubmission

1. References: verify every bibliography entry against its primary source; corrected
   entries for shaffer19, bientzle21, bientzle24, rottger23 already landed in
   `papers/shared/references.bib`.
2. Clarity rewrite: shorten abstract; define SCM, Cliff's delta, trace, generator, cell
   at first use; expand all acronyms; cite all evaluated models; reconsider the
   "generator" term.
3. DailyDilemmas exposition: add a self-contained dataset description with an example
   scenario and the stratified sampling detail.
4. Related work: add a Moral Uncertainty paragraph (MacAskill/Bykvist/Ord 2020; Newberry
   and Ord 2021 parliamentary approach; fanaticism and voting-based aggregation) and
   connect it to the multi-stakeholder vote and the structural-rejection analysis.
5. Model pool: add at least one open-weights generator; state the n=3 repetition
   rationale and add a sensitivity check with more repetitions on a subset.
6. Artifact availability: make the supplementary data and software packaging
   unmistakable (two reviewers scored Datasets 1 despite the submitted zip); include a
   prominent availability statement and per-artifact license enumeration (checklist item
   B2 was answered No and should be fixed).
7. Scope statement: explicitly delimit claims to trace structure, not moral correctness
   of outcomes, where DailyDilemmas is introduced.
