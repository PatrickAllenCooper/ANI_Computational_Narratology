# Method Experimentation Lab

Single environment for querying, evaluating, and comparing every prompt/protocol variant defined in this repo. Entry point: `method_lab.ipynb`.

Canonical study design: `Guidance_Documents/study_design.md`. This lab does not invent new methods; it surfaces existing ones for experimentation (especially sycophancy evaluation on ELEPHANT and related datasets).

## Top 5 for a robust study

The full inventory below is large. If you only have time to run one comparison well, run these five — they are the minimal set the project's own findings say a robustness claim depends on. All five are also flagged `recommended=True` in the notebook registry.

| Variant | Role | Why this one |
|---------|------|--------------|
| `narrative_cot` | Base scaffold | Hand-designed NoT; every main-paper structural claim (stakeholder count, uncertainty score) anchors here. Any variant must be compared back to this. |
| `standard_cot` | Required control | The mandatory negative control — without it, no effect size is interpretable. |
| `narrative_cot_v3` | Best generalizing optimized variant | Cross-family textual-gradient result; `v2` overfits its training judge family (bigger generosity gap), `v3` closes/inverts that gap and is the variant integrated into the main paper. |
| `phase18_robust` | Best robustness-audited optimized variant | Panel-robust optimizer; explicitly built to survive held-out-judge and human-gold evaluation, unlike `sg_narrative_grad` which Goodharts its single training judge. |
| `textgrad_cot` | Honesty check on mechanism | The "best-optimized-CoT" comparator. Phase 11 shows N-CoT beats it decisively — any robustness study should re-verify this rather than assume it. |

## Variant inventory

| Variant | Kind | Provenance |
|---------|------|------------|
| `raw` | static | Phase 1 / ELEPHANT bare user turn |
| `baseline_io` | static | Phase 1 direct answer |
| `standard_cot` | static | Phase 1 step-by-step |
| `standard_cot_verbose` | static | E1 matched-budget length control |
| `standard_cot_refusal_tuned` | static | E3 safety-wrapper control |
| `narrative_cot` | static | Hand NoT (main paper) |
| `narrative_cot_v2` | static | Phase 10 in-family textual gradient |
| `narrative_cot_v3` | static | Phase 10b cross-family (= `ng2` final) |
| `drop_protagonist` … `drop_commitment` | ablation | Phase 2 section knockouts (5) |
| `textgrad_cot` | optimized | Phase 9/11 TextGrad on standard CoT |
| `sg_narrative_grad` | optimized | Phase 14 single-judge sycophancy gradient |
| `sg_textgrad_cot` | optimized | Phase 14 TextGrad baseline |
| `sg_opro` / `sg_ape` | optimized | Phase 14 OPRO / APE baselines |
| `phase18_robust` | optimized | Phase 18 panel-robust (submission gate) |
| `debate_not` / `debate_std_cot` / `debate_textgrad_cot` | protocol | Multi-agent R0–R4 debate arms |

## Notebook sections

1. Setup (repo root, imports, `.env`)
2. Variant registry (built programmatically from `scripts/` + summary JSONs)
3. Query one item (live API; gated)
4. Batch evaluation on ELEPHANT / BrokenMath (smoke defaults; full CLI recipes)
5. Existing committed results + recreation commands
6. Cost and cache notes

## Datasets

- ELEPHANT: `scripts/load_elephant.py` (raw under `data/elephant/`, gitignored; downloads on first use).
- BrokenMath: `scripts/load_brokenmath.py` (`data/brokenmath/`, gitignored).
- DailyDilemmas: via `scripts/run_phase1_quartet.load_daily_dilemmas` (HuggingFace `datasets`).

## Cost

Smoke cells default to `n=1` or `n=10`. Full-quartet grids are expensive; prefer CLI runners with `--smoke` first. Per-cell caches live under `divergence_study_outputs/` (gitignored patterns in `.gitignore`).
