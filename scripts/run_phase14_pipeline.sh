#!/usr/bin/env bash
# Phase 14 full pipeline: all optimizers, holdout, aggregate.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "=== Phase 14 narrative grad ==="
python -m scripts.run_phase14_syco_grad

echo "=== Phase 14 TextGrad CoT ==="
python -m scripts.run_phase14_tg_cot

echo "=== Phase 14 OPRO ==="
python -m scripts.run_phase14_opro

echo "=== Phase 14 APE ==="
python -m scripts.run_phase14_ape

echo "=== Phase 14 holdout + aggregate ==="
python -m scripts.run_phase14_holdout
python -m scripts.aggregate_phase14

echo "=== ELEPHANT v2/v3 (if not done) ==="
python -m scripts.run_elephant \
  --arms narrative_cot_v2,narrative_cot_v3 \
  --datasets oeq,aita_yta,ss,flip_pairs --n 150 --workers 6

python -m scripts.aggregate_elephant
python -m scripts.make_sycophancy_charts

echo "Phase 14 pipeline complete."
