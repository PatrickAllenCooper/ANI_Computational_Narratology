---
artifact_contract: "ce-handoff/v1"
created_at: "2026-09-22T00:00:00Z"
title: "Handoff for a new chat: designing the results section of the unified NoT paper"
summary: "All experimental validation of pillars 1 and 3 is complete and registered. Both headline claims (routing beats the collective; sycophancy reduction replicates cross-vendor) are statistically solid as scoped. The counterparty mechanism was tested causally twice (section knockout, isolated instruction) and failed both times, the second time reversing direction; this is folded into the paper honestly. The open task is DESIGN: papers/unified/ compiles at 23 pages under an ACL long-paper style, roughly 3x a typical page budget, driven almost entirely by results_pillar3.tex (~4,100 words), a results section imported wholesale from a separate paper and never cut down to one pillar's proportionate share. This document hands that design task to a fresh session with full context."
keywords: ["handoff", "results section design", "papers/unified", "pillar 1", "pillar 3", "Addendum 17", "counterparty causal test", "page budget"]
cwd: "/Users/pat/code/ANI_Examination"
branch: "sycophancy-scenario-redesign"
head: "e7ccf8c"
resume_focus: "Design and cut papers/unified/results.tex (and its imported results_pillar3.tex) down to a proportionate, well-organized results section for a two-pillar paper. All the evidence is validated and registered; nothing here requires new experiments unless this document's 'genuinely still open' section says otherwise."
repository: "ANI_Examination"
branch_full: "sycophancy-scenario-redesign"
---

# Handoff, 2026-09-22: designing the results section

This is a **task handoff into a fresh chat**, not a status report to read and discard.
Whoever picks this up should treat the "what's settled" section as fact (do not
re-derive it), treat "what's genuinely open" as the only place new experiments might be
warranted, and treat "the design task" as the actual work to do next.

Read this document in full before touching `papers/unified/results.tex`. The prereg
(`Guidance_Documents/prereg_embodiment_community.md`, now ~7,400 lines) is the
authoritative source for every number; this document indexes it, it does not replace it.

---

## 1. Standing rules (unchanged; apply to everything below)

1. **Register before spend.** Every experiment gets an addendum in the prereg with a
   ceiling, stop rule, readouts with baselines, pre-declared readings for every branch,
   and guards, BEFORE any model call. Results are appended as `## X RESULTS` blocks.
   Nothing above an appended block is ever edited.
2. **Read mechanically. The guard is a literal string.** `GUARD FAILED` printed anywhere
   means the numbers are unread. Under-gated readouts (fewer than the registered minimum
   fired, usually 60) are reported as under-gated, never tabulated.
3. **Caches are keyed by NAME.** A new arm or instrument gets a new name; a script that
   silently falls back to a default name/instrument when given a non-default one is a
   landmine (two such bugs were found and fixed this session, see §4).
4. **Every citation is fetched at a primary page** (the prior ARR rejection was for
   fabricated references). Ledger: `papers/embodied_sensor/related_works_verified.json`.
5. **Prose style, strictly enforced:** no em dashes, no colons or semicolons in running
   prose, no lists in the paper text (this handoff itself is a document, not the paper,
   so it can use lists). Every number carries a `%` source comment pointing at a prereg
   line or an artefact key. Every `.tex` file has a comment-free `*_clean.tex` twin,
   regenerated with `scripts/make_clean_tex.py`, never hand-edited.
6. **PI's standing instruction (repeated and reaffirmed across this whole programme):
   "don't argue a stronger case than what we can directly justify."** This is the single
   most important test to apply while designing the results section. Several sentences
   were caught overclaiming this session (see §4) and fixed; assume more may be lurking
   in the imported pillar-3 text, which was written for a different, standalone paper.
7. **Pillar 2 (theory of mind) is out of scope**, PI decision 2026-09-21. It is a
   `\todo`-marked skeleton in every file (abstract, introduction, theory, methods,
   results) and must stay that way. Do not expand it, do not run Addendum 18 stage 2,
   do not let it creep back into the abstract or introduction as a claim.
8. **n-player games are future work** (older PI instruction, still standing).
9. Attribution for commits from this session: `Co-Authored-By: Claude Sonnet 5
   <noreply@anthropic.com>` (the model changed mid-programme from Opus 5; use whichever
   line your own session's system reminder gives you, don't copy this one blindly).

---

## 2. What's settled (statistically solid, no new experiments needed for these)

### 2a. Pillar 3: routing beats the collective

**Claim:** reviewing cases flagged by a second dissenting seat with a stronger,
uninterested judge raises accuracy over the collective's own verdict alone.

**Evidence, all independently reproduced by a from-scratch second script sharing no
code with the registered analyzers (`scripts/verify_pillar3_headline.py`):**

| condition | debates | collective acc | routed acc | delta [95% CI] | source |
|---|---|---|---|---|---|
| grok, on-edge (registered) | 1,677 | 0.8426 | 0.9129 | +0.0704 [+0.0477, +0.0941] | 16.15.1/16.22; `router_decomposition.json` |
| grok, edges cut (no exchange between seats) | 1,675 | 0.8322 | 0.8800 | +0.0478 [+0.0316, +0.0655] | 16.25 RESULTS; `router_decomposition_noedge_k4.json` |
| haiku-249 (low role-lock, 0.42) | 487 | 0.8542 | 0.8912 | +0.0370 [+0.0145, +0.0593] | 16.23 RESULTS; `router_decomposition_haiku.json` |
| Llama-3.3-70B-Instruct (open-weight, third vendor) | 897 | 0.6198 | 0.6611 | +0.0412 [+0.0278, +0.0567] | 16.24 RESULTS; `router_decomposition_llama.json` |

Never failed a CI-excludes-zero check across four structurally different conditions
(a topology ablation, two very different role-lock regimes, three vendors). This is the
strongest-evidenced result in the whole programme. **Ship as written.**

**The narrower, correctly-scoped companion claim** ("+2.4 points over the strongest
single model," 16.22 R1) holds ONLY on grok (+0.0238 [+0.0048, +0.0452]) and explicitly
does not replicate on Llama (+0.0045 [0.000, +0.0112], CI includes zero). Diagnosed this
session (see §4): Llama's own solo model (84.2% accuracy) vastly outperforms its own
three-seat deliberation collective (62.0%, a 22-point gap, versus grok's 7-point gap),
AND the certified flag fires six times more rarely on Llama (2.0% of cells vs grok's
12.1%). The abstract already scopes this correctly ("on the model whose seats lock to
their assigned side... does not replicate on a model whose seats do not"). **Keep the
scoping; do not generalize this sentence.**

### 2b. Pillar 1: cross-vendor sycophancy reduction

**Claim:** the scaffold reduces validation of the asker's own account, replicating
across vendors including open-weight models.

**Evidence:**

| model | vendor | drop [95% CI] | non-response (CoT/NoT) | source |
|---|---|---|---|---|
| claude-haiku-4-5 | Anthropic | -26.7 [-36.7, -16.7] | 0% / 0% | 16.15.7 |
| claude-sonnet-4-6 | Anthropic | -22.3 [-31.3, -13.3] | 0% / 1% | 16.15.7 |
| gpt-5.4-nano | OpenAI | -41.4 [-51.1, -31.4] | 3% / 23% | 16.15.7 (nano's attrition is the known weak point) |
| grok-4-1-fast-reasoning | xAI | -36.0 [-44.9, -27.1] | 3% / 3% | 16.15.7 |
| Llama-3.3-70B-Instruct | Meta (open-weight) | **-69.9** [-77.7, -61.4] | 4% / 6% | 17.1 RESULTS |
| Mistral-Large-3-2 | Mistral (open-weight) | **-58.2** [-66.2, -49.7] | 5% / 7% | 17.1 RESULTS |
| DeepSeek-V4-Pro | DeepSeek (open-weight) | **-45.5** [-53.8, -37.1] | 4% / 6% | 17.1 RESULTS |

Sanctioned phrasing: **22 to 70 points absolute, 31 to 87 percent relative, seven models
from six vendors, three of them open-weight** (`pillar1_headline.json`). The open-weight
deployments show the largest effects and the cleanest attrition profile of any cell.
**Ship as written.**

**Judge-robustness (17.2):** a second judge (gpt-5.4-nano) confirms direction and
significance on 6 of 7 models (sonnet's CI touches zero: -6.9 [-14.9, +1.1]). A third
judge (Llama-3.3-70B-Instruct) reverses sign with a CI excluding zero on 3 of 7 (haiku,
sonnet, nano all go POSITIVE), traced to that judge scoring "validates" on 80-100% of
responses regardless of content (an acquiescence defect in that specific judge, not
evidence against the effect; validation rate is pinned near ceiling on almost every
cell). This is reported honestly in the paper as a named, diagnosed exception, not
smoothed over. **Ship as written; do not claim "judge-robust" without the qualifier.**

---

## 3. What's genuinely NOT settled (tested and failed; do not re-test the same way)

**The counterparty-representation mechanism does not survive as a causal, standalone
claim, tested twice this session, both times negative:**

1. **17.3 (section knockout):** remove one of NoT's three candidate sections
   (stakeholders, consequences, uncertainty) at a time. Pooled across haiku/grok/nano, no
   single section's removal retakes a quarter of the reduction with a CI excluding zero
   (NEITHER branch). Model-heterogeneous: on haiku every section matters roughly equally
   (0.42-0.55 share), on grok and nano almost none of them matter (near-zero share, wide
   CIs). Does not discriminate the counterparty account from the competing
   hedging-register account.
2. **17.5 (isolated instruction, run today):** add ONE sentence ("consider everyone
   affected... not only the person asking") to plain standard CoT, no other NoT
   structure, with a style-matched control (a premise-check sentence, same length,
   unrelated content). **This REVERSES: pooled +6.1 [+2.1, +10.2], the instruction
   RAISES validation.** None of the three pre-declared branches anticipated this; the
   analysis script (`scripts/analyze_counterparty_causal.py`) now has an explicit
   REVERSED branch added after the fact, documented as a registration gap, not silently
   absorbed.

**What survives:** the purely observational, item-level finding (Z2) that the
naturally-occurring reduction is three times larger on accounts that already name a
second party (-37.7 vs -11.3 points, diff -26.3 [-35.7, -17.0], holds on 3 of 4 original
models). This is a real, well-powered correlation. It is not, and per the above cannot
currently be claimed as, a causal mechanism.

**The abstract, theory.tex, theory.md and results.tex all now say this precisely**
("an association rather than a manipulated cause"; "the item-level association... and
the causal, instruction-level claim... are different claims, and only the first is
supported"). **Do not strengthen this language without new evidence, and do not weaken
it either, it is exactly calibrated to what two negative causal tests support.**

**If a future session wants to actually find a causal mechanism**, the two tested
designs (remove-a-section, add-an-isolated-instruction) both failed, which itself is
informative: the reduction may depend on structural interaction between sections
(counterparty-awareness plus consequence-tracing plus deferred commitment together)
rather than any one part. A design that manipulates a PAIR of sections at once
(add counterparty-awareness AND consequence-tracing, withhold the rest) would be the
next thing to try, not a repeat of either existing design. This is a real open research
question, not a paper-writing task, and should not be undertaken casually inside a
"design the results section" session without the PI deciding it's worth the spend.

---

## 4. What was found and fixed this session (context for reading the git log)

Four real bugs, all the same underlying pattern (a script silently defaulting instead of
using what it was actually given), all found by hand-computing a ground truth from raw
CSVs and noticing the script's output didn't match:

1. `scripts/analyze_flooding.py`: `_aita_block` hardcoded the default `AITA` instrument
   even when a differently-seat-named instrument (`AITA_NOEDGE`) had built the debates
   correctly, silently zeroing every phi and objection-rate number for the edges-off
   cell. Fixed, regression test added, registered grok cell reproduces byte-identically
   before and after.
2. `scripts/verify_pillar3_headline.py` (my own verification script): the same
   seat-naming bug in its own diagnostic cross-check. Cosmetic only (the real numbers
   never depended on it), fixed anyway.
3. `scripts/rescore_elephant_untruncated.py`: `_report()` crashed on a custom `--arms`
   list that excludes the default arms (exactly the case for every knockout/causal-test
   rescore this session). Fixed; no spend was lost since scores were already cached.
4. A broken LaTeX cross-reference (`\ref{sec-results-voter}`, no such label) that I
   introduced myself while writing the new 16.17-16.21 subsection. Found via the build
   log's "undefined references" warning, fixed to point at the real label
   (`sec-results-collective`).

Two overclaiming sentences were caught and fixed in the abstract (the counterparty
association stated as fact with no causal hedge; "beats the strongest single model"
stated as if general when it's grok-only). **Read every sentence in
`papers/unified/*.tex` with the same suspicion before calling the design pass done.**

---

## 5. The design task itself

### 5a. The core problem, quantified

`papers/unified/main.tex` builds cleanly (23 pages, `review` ACL style). That is
**roughly 3x a typical ACL/ARR long-paper body limit** (8-9 pages plus references is
standard; even a generous target would not want 23). Word counts by section:

| file | words | share |
|---|---|---|
| `results.tex` (pillar 1 + pillar 2 skeleton) | 1,192 | |
| `results_pillar3.tex` (pillar 3, imported) | **4,073** | the dominant driver |
| `theory.tex` | 1,379 | |
| `methods.tex` (pillar 1 + 2 skeleton) | 773 | |
| `methods_collective.tex` (pillar 3, imported) | 1,899 | |
| `introduction.tex` | 824 | |
| everything else | ~550 | |

`results_pillar3.tex` and `methods_collective.tex` were derived mechanically from
`papers/embodied_sensor/` (a SEPARATE, standalone ICLR-targeted paper about the
collective alone) by a sed script that demotes `\section` to `\subsection` and
`\subsection` to `\subsubsection`. That paper's results section was appropriately sized
for a paper whose ENTIRE content is the collective. In the unified two-pillar paper it is
one of two pillars and is currently about 3.4x the length of pillar 1's results, which
is not a proportionate allocation for a paper that wants to make both claims well.

**This is the actual design task**: decide what pillar 3's results section needs to say
to support ONLY the claims in §2a above (plus the honestly-scoped exceptions), and cut
the rest to what belongs in an appendix or can be dropped. The embodied_sensor source
(`papers/embodied_sensor/results.tex`) covers material the unified paper doesn't need at
this depth: the full roles-by-edges 2x2 table and all four extent-of-method boundary
cases (silence, flooding, no-counterparty, coordination games) were written for a paper
whose whole point was mapping the sensor's boundaries. The unified paper's point is
narrower (does routing help, does the scaffold reduce sycophancy); it needs enough of
that material to source the headline numbers and nothing more.

### 5b. Concrete design questions to resolve

1. **How much of `results_pillar3.tex`'s five subsubsections does the unified paper
   actually need?** Candidate cut: keep "The collective has one effective voter" (short,
   motivates why routing is needed at all) and "The dissent is an error sensor"
   (the mechanism itself, where the headline numbers live) at close to full length;
   compress "Embodiment places the bias," "Who decides changes the verdict," and
   "The extent of the method" to one paragraph each or move to an appendix.
2. **Does pillar 1 need a figure?** It currently has zero (a table only,
   `pillar1_headline.tex`). `scripts/make_sycophancy_charts.py` exists from the earlier
   sycophancy paper and could be adapted to `pillar1_headline.json`, three bars per
   generator or similar. Pillar 3 has five figures already (fig1, fig3, fig4, fig5, fig6,
   plus the mechanism figure A/B/C in methods). A results section this asymmetric in
   figure count is itself a design signal.
3. **Voice consistency.** Pillar 1's prose (written fresh this session, in
   `results.tex`) and pillar 3's prose (imported, written by an earlier session for a
   different paper) have slightly different rhythms. Read them back to back before
   deciding whether to rewrite pillar 3's kept material in pillar 1's voice or leave it,
   given the derivation pipeline (`sed` script in `papers/embodied_sensor/results.tex` ->
   `results_pillar3.tex`) means hand-editing the derived file directly will be
   overwritten the next time someone regenerates it from the source. **Any content cut
   or rewrite for pillar 3 belongs in `papers/embodied_sensor/results.tex` first, then
   regenerate `results_pillar3.tex`** with:
   ```
   cd papers/unified
   sed -e 's/\\section{Results}/\\subsection{Pillar 3, the collective}/' ../embodied_sensor/results_clean.tex \
     | sed -e '/subsection{Pillar 3, the collective}/!s/\\subsection{\([^}]*\)}/\\subsubsection{\1}/' > results_pillar3.tex
   cd ../.. && .venv/bin/python -m scripts.make_clean_tex papers/unified/results_pillar3.tex
   ```
   **If the cut is severe enough that mechanical demotion no longer makes sense** (for
   example, folding pillar 3's results into one un-subdivided subsection to match
   pillar 1's style), abandon the derivation pipeline for that file and write
   `results_pillar3.tex` directly, noting in its header comment that it is no longer
   auto-derived and why.
4. **Where do the extent-of-method boundary cases (silence, flooding, no-counterparty,
   coordination games) belong?** They are true, well-registered, and directly support
   the paper's honesty about where the mechanism does and doesn't work, but a short
   paper making two focused claims may not need the full map. Consider one sentence in
   results plus the full table in an appendix, rather than the current full prose
   treatment.
5. **The `results.tex` \S1 pillar-1 discussion is already fairly tight (1,192 words
   covering: the headline table, the verbose/length-matched controls, the counterparty
   association AND its two causal failures, the seven-model replication, the
   judge-dependence finding, the section-knockout NEITHER finding, the verdict-instrument
   scope boundary).** This is close to a reasonable proportionate length already; the
   main lever is pillar 3, not pillar 1.
6. **Check the ACL page target explicitly** (ask the PI which venue/track this is
   actually aiming for now that ARR/ACL is the stated fallback per earlier handoffs) and
   design to that number, not to an assumed 8 pages. `main.tex` currently uses
   `\usepackage[review]{acl}`; check whether the venue wants `review` (double-column,
   anonymous) or a different mode before finalizing.

### 5c. Do NOT touch these while doing the design pass

- `papers/embodied_sensor/*.tex` beyond the source edits needed to regenerate
  `results_pillar3.tex`/`methods_collective.tex` correctly. That is a separate paper
  with its own ICLR 2027 target; changes there should be deliberate, not incidental to
  trimming the unified paper.
- Any number. This pass is about structure, length, and voice, not re-deriving or
  re-wording claims (beyond fixing genuine overclaims per §6 below, which is about
  honesty, not design).
- Pillar 2's skeleton status (§1 rule 7).

---

## 6. A standing check to run before considering the results section "designed"

Grep every sentence in the final `results.tex` / `results_pillar3.tex` against the
prereg block it cites. Two categories of error to hunt for, both found and fixed once
already this session:
1. **A correlational finding stated with causal language** (the counterparty sentence
   pattern from §4).
2. **A model-specific finding stated as if general** (the "beats strongest model"
   pattern from §4).

A fast way to re-check: every number in the paper has a `%` source comment naming a
prereg addendum or artefact key (rule 5 in §1). Pull each cited block from the prereg
and confirm the sentence around the number doesn't say more than the block's own
"reading" paragraph says.

---

## 7. File index

- `papers/unified/main.tex` — top-level document, `\input`s everything else in order:
  abstract, introduction, theory, methods, results, related_work, limitations.
- `papers/unified/abstract.tex`, `introduction.tex`, `theory.tex`, `methods.tex`,
  `results.tex`, `limitations.tex`, `related_work.tex` — written fresh this programme,
  cover both pillars (pillar 2 as a skeleton throughout).
- `papers/unified/methods_collective.tex`, `results_pillar3.tex` — DERIVED files, see
  §5b(3) for the regeneration command. Do not hand-edit.
- `papers/unified/theory.md` — the design document behind `theory.tex`, has the full
  prediction-to-experiment table (`P1a`-`P1c`, `P3a`-`P3f`), all rows now closed out or
  explicitly marked open.
- `papers/unified/pillar1_headline.tex` — the pillar-1 headline table, generated by
  `scripts/make_pillar1_table.py` from `divergence_study_outputs/pillar1_headline.json`.
  Regenerate, don't hand-edit.
- `papers/unified/build.sh` — `cd papers/unified && ./build.sh` to rebuild
  (`latexmk`, shared style from `papers/shared/`).
- `papers/unified/*_clean.tex` — comment-free twins, regenerate with
  `python -m scripts.make_clean_tex <file>.tex` after any edit to the commented source.
- `Guidance_Documents/prereg_embodiment_community.md` — the full experimental record,
  ~7,400 lines. Search for `Addendum 17` for everything from this session (17.1 through
  17.5), `Addendum 16.23`/`16.24`/`16.25` for the pillar-3 four-condition validation,
  `Z2` through `Z5` for the zero-spend analysis blocks.
- `Guidance_Documents/HANDOFF_2026-09-18.md`, `HANDOFF_2026-09-21.md` — prior handoffs,
  still accurate for anything not superseded by this document or the prereg.
- Analysis scripts written this session, all offline-testable (`--selftest`):
  `scripts/analyze_counterparty_moderator.py` (Z2), `scripts/analyze_bridge_shielding.py`
  (Z4), `scripts/make_validation_judge_reliability.py` (Z5),
  `scripts/make_pillar1_table.py` (pillar-1 headline table, now 7-model),
  `scripts/analyze_knockout_social.py` (17.3), `scripts/analyze_counterparty_causal.py`
  (17.5), `scripts/verify_pillar3_headline.py` (independent pillar-3 recomputation, 4
  conditions), `scripts/analyze_tom_mechanism.py` (pillar-2 diagnostic, informational
  only, pillar 2 is out of scope for the paper).

---

## 8. Fully separate, deferred items (not part of the design task, listed for completeness)

- **17.4** (proper length-matched control, gated at $10, originally gated on $95
  measured spend across 17.1-17.3): never launched. Exact spend was never fully
  reconstructed (generation-only estimate was ~$7 in completion tokens alone, missing
  prompt-token and judge-scoring cost). Decide with an exact accounting before launching,
  or explicitly close Addendum 17 without it.
- **16.25's two still-under-gated cells** (stdcot k=4, n-perspective k=2): registered,
  never launched this session. Commands unchanged from `HANDOFF_2026-09-18.md` §6.1. The
  xAI queue is free now.
- **Figure regeneration**: none of the existing figures were touched this session despite
  new no-edge-k4 and Llama numbers landing. `make_mechanism_figure.py --check` and
  `make_graph_figures.py --check` should be re-run if those figures are kept in the
  design pass's cut version of the paper.
- **Optional tier** (Mistral/DeepSeek communities, gpt-4o, sonnet-249 community): entirely
  undecided, not discussed this session.
