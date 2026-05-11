"""
scripts/make_kc_figure.py -- Quick K_C proxy correlation bar chart from
phase 4 outputs.  No additional API calls.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path("./divergence_study_outputs")


def main() -> int:
    csv = OUT_DIR / "kc_proxy_correlation.csv"
    if not csv.exists():
        print(f"No correlation CSV at {csv}")
        return 1
    df = pd.read_csv(csv)
    df = df[df["generator"] != "gpt-4o"].copy()
    if df.empty:
        return 1

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    gens = df["generator"].tolist()
    rhos = df["spearman_rho"].astype(float).tolist()
    ns = df["n"].astype(int).tolist()
    bars = ax.bar(range(len(gens)), rhos,
                  color=["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"][:len(gens)])
    for i, (b, n) in enumerate(zip(bars, ns)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                f"n={n}", ha="center", fontsize=8)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_xticks(range(len(gens)))
    ax.set_xticklabels([g.replace("-", "\n", 1) for g in gens], fontsize=8)
    ax.set_ylim(-0.1, 1.0)
    ax.set_ylabel(r"Spearman $\rho$ (gzip K_C vs N-CoT direction)", fontsize=9)
    ax.set_title(r"K_C gzip proxy tracks the N-CoT structural shift", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = OUT_DIR / "kc_proxy_correlation.pdf"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
