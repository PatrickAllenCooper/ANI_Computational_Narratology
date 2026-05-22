"""scripts/make_hero_figure.py

Generate the hero schematic for ACL_paper.tex (papers/figures/hero_mechanism.pdf).

The figure communicates the mechanism intuition in one glance:
* left panel: standard CoT runs a linear reasoning chain over the dilemma and
  exhibits stakeholder collapse + uncertainty suppression at the head of the
  output;
* right panel: N-CoT scaffolds the same input through five narrative
  primitives (protagonist, stakeholders, consequences, uncertainty,
  commitment) before any commit, raising structural variables and dropping
  the two empirically firing failure modes;
* footer strip: the multi-stakeholder extension reuses N-CoT as the per-agent
  generator and adds a moderator-integrated proposal plus binary vote.

The palette is the project-wide one defined in figure_style.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_style import (  # noqa: E402
    apply_paper_style,
    TEXT_WIDTH,
    CONDITION_COLORS,
    GENERATOR_COLORS,
)

OUT_LOCAL = Path("divergence_study_outputs/hero_mechanism.pdf")
OUT_PAPER = Path("papers/figures/hero_mechanism.pdf")


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    fc: str,
    ec: str = "#3F3F46",
    tc: str = "#0F172A",
    fs: float = 8.0,
    weight: str = "normal",
    rounding: float = 0.6,
    ha: str = "center",
    va: str = "center",
    lw: float = 0.7,
) -> None:
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.05,rounding_size={rounding}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    tx = x + w / 2 if ha == "center" else (x + 0.4 if ha == "left" else x + w - 0.4)
    ty = y + h / 2 if va == "center" else (y + 0.4 if va == "bottom" else y + h - 0.4)
    ax.text(
        tx,
        ty,
        text,
        ha=ha,
        va=va,
        fontsize=fs,
        color=tc,
        fontweight=weight,
    )


def arrow(ax, x0: float, y0: float, x1: float, y1: float, *,
          color: str = "#475569", lw: float = 1.0,
          style: str = "-|>") -> None:
    a = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=style,
        mutation_scale=10,
        linewidth=lw,
        color=color,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(a)


# ---------------------------------------------------------------------------
# Main composition
# ---------------------------------------------------------------------------

def main() -> None:
    apply_paper_style()

    fig = plt.figure(figsize=(TEXT_WIDTH, 4.6))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70)
    ax.set_axis_off()

    std = CONDITION_COLORS["standard_cot"]
    ncot = CONDITION_COLORS["narrative_cot"]
    pale_std = "#E2E8F0"
    pale_ncot = "#CFE3E2"

    # ----- title strip -----
    ax.text(
        50, 67.5,
        "Standard CoT collapses the dilemma into a single chain; "
        "Narrative CoT scaffolds reasoning through five primitives of narrative",
        ha="center", va="center",
        fontsize=10.0, fontweight="semibold", color="#0F172A",
    )

    # ----- left panel: Standard CoT -----
    L_x, L_w = 3, 44
    L_top = 63

    box(
        ax, L_x, L_top - 4, L_w, 3.2,
        "Standard Chain-of-Thought",
        fc=std, ec=std, tc="white", fs=9.5, weight="semibold",
        rounding=0.5,
    )

    # Input
    box(ax, L_x + 12, L_top - 9.5, L_w - 24, 3.2,
        "Dilemma (DailyDilemmas item)",
        fc="#F8FAFC", ec="#94A3B8", fs=8.0)
    arrow(ax, L_x + L_w / 2, L_top - 9.7, L_x + L_w / 2, L_top - 12, color="#64748B")

    # Linear chain
    chain_top = L_top - 12
    for i, step in enumerate([
        "step 1: weigh competing factors",
        "step 2: pick the better option",
        "step 3: commit to the answer",
    ]):
        y = chain_top - 3 - i * 4.6
        box(ax, L_x + 6, y, L_w - 12, 3.4, step,
            fc=pale_std, ec=std, fs=8.0)
        if i < 2:
            arrow(ax, L_x + L_w / 2, y, L_x + L_w / 2, y - 1.2,
                  color="#64748B")

    # Linear answer
    arrow(ax, L_x + L_w / 2, L_top - 28.5, L_x + L_w / 2, L_top - 30.5,
          color="#64748B")
    box(ax, L_x + 6, L_top - 34, L_w - 12, 3.4,
        "Terse answer (single recommendation)",
        fc="white", ec=std, fs=8.0)

    # Failure callouts inside the panel frame
    callout_top = L_top - 37.5
    ax.text(L_x + 2, callout_top, "Empirically fires on this scaffold:",
            ha="left", va="top", fontsize=8.0, color="#B53A2E",
            fontweight="semibold")
    ax.text(L_x + 3.5, callout_top - 3.2,
            r"$\bullet$  stakeholder collapse  (15--31% of items)",
            ha="left", va="top", fontsize=7.8, color="#B53A2E")
    ax.text(L_x + 3.5, callout_top - 6.0,
            r"$\bullet$  uncertainty suppression  (50--72% of items)",
            ha="left", va="top", fontsize=7.8, color="#B53A2E")

    # Panel border (subtle)
    panel_l = FancyBboxPatch(
        (L_x - 1, L_top - 50), L_w + 2, 49,
        boxstyle="round,pad=0.1,rounding_size=0.8",
        linewidth=0.4, edgecolor="#CBD5E1", facecolor="none",
    )
    ax.add_patch(panel_l)

    # ----- right panel: N-CoT -----
    R_x, R_w = 53, 44
    R_top = 63

    box(
        ax, R_x, R_top - 4, R_w, 3.2,
        "Narrative Chain-of-Thought (N-CoT)  --  prompt-only",
        fc=ncot, ec=ncot, tc="white", fs=9.5, weight="semibold",
        rounding=0.5,
    )

    box(ax, R_x + 12, R_top - 9.5, R_w - 24, 3.2,
        "Same dilemma",
        fc="#F8FAFC", ec="#94A3B8", fs=8.0)
    arrow(ax, R_x + R_w / 2, R_top - 9.7, R_x + R_w / 2, R_top - 12,
          color="#64748B")

    sections = [
        ("1. Protagonist", "name, role, what they know"),
        ("2. Stakeholders", "parties intersecting the decision"),
        ("3. Consequences", "$\\geq 2$ steps forward for each action"),
        ("4. Uncertainty", "what remains genuinely unknown"),
        ("5. Commitment", "chosen action + narrative warrant"),
    ]
    sec_top = R_top - 12
    sec_h = 3.0
    sec_gap = 0.45
    for i, (title, desc) in enumerate(sections):
        y = sec_top - 2.9 - i * (sec_h + sec_gap)
        box(ax, R_x + 2, y, 11, sec_h, title,
            fc=ncot, ec=ncot, tc="white", fs=7.5, weight="semibold")
        box(ax, R_x + 13.5, y, R_w - 15.5, sec_h, desc,
            fc=pale_ncot, ec=ncot, fs=7.6)

    last_y = sec_top - 2.9 - 4 * (sec_h + sec_gap)
    arrow(ax, R_x + R_w / 2, last_y, R_x + R_w / 2, last_y - 1.6,
          color=ncot)
    box(ax, R_x + 6, last_y - 5.0, R_w - 12, 3.4,
        "Narratively warranted decision",
        fc=ncot, ec=ncot, tc="white", fs=8.0, weight="semibold")

    callout_top_r = last_y - 7.5
    ax.text(R_x + 2, callout_top_r,
            "Structural variables rise; failure modes drop:",
            ha="left", va="top", fontsize=8.0, color=ncot,
            fontweight="semibold")
    ax.text(R_x + 3.5, callout_top_r - 3.2,
            r"$\bullet$  stakeholder count rises, collapse $\to$ ${<}1$%",
            ha="left", va="top", fontsize=7.8, color=ncot)
    ax.text(R_x + 3.5, callout_top_r - 6.0,
            r"$\bullet$  uncertainty score rises, suppression $\to$ 1--24%",
            ha="left", va="top", fontsize=7.8, color=ncot)

    panel_r = FancyBboxPatch(
        (R_x - 1, R_top - 50), R_w + 2, 49,
        boxstyle="round,pad=0.1,rounding_size=0.8",
        linewidth=0.4, edgecolor="#CBD5E1", facecolor="none",
    )
    ax.add_patch(panel_r)

    # ----- footer strip: multi-stakeholder extension -----
    foot_top = 13.5
    ax.text(
        50, foot_top,
        "Multi-stakeholder extension (Section 5): three narrators each run N-CoT, then the "
        "moderator integrates their modification requests into a single proposal that each votes on.",
        ha="center", va="center",
        fontsize=8.2, color="#0F172A",
    )

    agent_specs = [
        ("Formal\ndecider", GENERATOR_COLORS["gpt-5.4-nano"]),
        ("Primary\naffected", GENERATOR_COLORS["claude-haiku-4-5"]),
        ("Third\nparty", GENERATOR_COLORS["claude-sonnet-4-6"]),
    ]

    # Compact arrangement: 3 small role boxes  ->  integrated proposal + vote  ->  consensus
    base_y = 2.5
    role_w, role_h = 5.5, 7.0
    role_gap = 1.0
    group_x = 2.5
    for i, (label, color) in enumerate(agent_specs):
        x = group_x + i * (role_w + role_gap)
        box(ax, x, base_y, role_w, role_h, f"N-CoT\n{label}",
            fc=color, ec=color, tc="white", fs=6.8, weight="semibold")
    end_roles = group_x + 3 * role_w + 2 * role_gap  # ~21.0

    # Stage 1 -> Stage 2 (proposal + vote)
    arrow(ax, end_roles + 0.5, base_y + role_h / 2,
          end_roles + 4.5, base_y + role_h / 2, color="#64748B")

    mid_x = end_roles + 5.5  # ~26.5
    mid_w = 28.0
    box(ax, mid_x, base_y + 1.3, mid_w, 4.4,
        r"moderator integrates proposal $\to$ binary vote",
        fc="white", ec="#94A3B8", fs=7.5)

    after_mid = mid_x + mid_w  # ~54.5

    # Stage 2 -> Stage 3 (consensus)
    arrow(ax, after_mid + 0.5, base_y + role_h / 2,
          after_mid + 4.5, base_y + role_h / 2, color="#64748B")

    cons_x = after_mid + 5.5  # ~60.0
    cons_w = 36.0
    box(ax, cons_x, base_y + 1.3, cons_w, 4.4,
        r"Defeasible consensus: $95\%$ full, $1.6\%$ rejection",
        fc=ncot, ec=ncot, tc="white", fs=7.6, weight="semibold")

    OUT_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_LOCAL, dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT_PAPER, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved {OUT_LOCAL} and {OUT_PAPER}")


if __name__ == "__main__":
    main()
