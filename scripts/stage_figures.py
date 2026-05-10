"""
scripts/stage_figures.py -- Copy generated figure PDFs from
divergence_study_outputs/ to papers/figures/ so tectonic can include them.

Run after any phase's aggregation step, before compiling the paper.

Usage:
  python -m scripts.stage_figures
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

OUT_DIR = Path("./divergence_study_outputs")
FIG_DIR = Path("./papers/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_MAP = {
    "failure_mode_firing_quartet.pdf": "failure_mode_firing_quartet.pdf",
    "tier1_effect_sizes_quartet.pdf":  "tier1_effect_sizes_quartet.pdf",
    "subinstruction_attribution.pdf":  "subinstruction_attribution.pdf",
    "debate_full_arc_quartet.pdf":     "debate_full_arc_quartet.pdf",
    "sycophancyeval_rates.pdf":        "sycophancyeval_rates.pdf",
    "agentic_probe_rates.pdf":         "agentic_probe_rates.pdf",
    "kc_proxy_correlation.pdf":        "kc_proxy_correlation.pdf",
}


def main() -> int:
    copied = 0
    missing = []
    for src_name, dst_name in FIGURE_MAP.items():
        src = OUT_DIR / src_name
        dst = FIG_DIR / dst_name
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Copied {src_name}")
            copied += 1
        else:
            missing.append(src_name)

    print(f"\nStaged {copied}/{len(FIGURE_MAP)} figures to {FIG_DIR}")
    if missing:
        print(f"Missing (not yet generated): {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
