"""scripts/rebuild_paper_figures.py -- Regenerate every paper figure
from the cached CSVs using the unified figure_style module.

This script does not call any LLM APIs. It only reads the CSVs in
divergence_study_outputs/ and writes PDFs that the paper includes.

Run order is independent; calling main() rebuilds everything.

Outputs (all under divergence_study_outputs/):

  failure_mode_firing_quartet.pdf   -- Section 4 headline figure
  tier1_effect_sizes_quartet.pdf    -- Section 4 effect sizes
  subinstruction_attribution.pdf    -- Section 4.1 ablation heatmap
  kc_proxy_correlation.pdf          -- Section 6 K_C proxy figure
  sycophancyeval_rates.pdf          -- Section 7 sycophancy panel
  agentic_probe_rates.pdf           -- Section 7 agentic panel
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

# Make the scripts directory importable when run as a plain script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_style import (
    apply_paper_style,
    annotate_bars,
    CONDITION_COLORS,
    CONDITION_LABELS,
    COL_WIDTH,
    GENERATOR_COLORS,
    GENERATOR_ORDER,
    panel_grid,
    percent_axis,
    save,
    short_generator_label,
    single_panel,
    style_generator_axis,
    TEXT_WIDTH,
    DELTA_NEGLIGIBLE,
    DELTA_SMALL,
    DELTA_MEDIUM,
)

OUT = Path("./divergence_study_outputs")


# ---------------------------------------------------------------------------
# Failure-mode firing rates (Section 4 headline)
# ---------------------------------------------------------------------------

def fig_failure_mode_firing() -> None:
    src = OUT / "failure_mode_firing_quartet.csv"
    if not src.exists():
        print(f"skip: missing {src}")
        return
    df = pd.read_csv(src)
    df = df[df["generator"] != "gpt-4o"].copy()
    modes = ["stakeholder collapse", "uncertainty suppression"]
    df = df[df["failure_mode"].isin(modes)].copy()
    if df.empty:
        return

    generators = [g for g in GENERATOR_ORDER if g in df["generator"].unique().tolist()]
    fig, axes, _ = panel_grid(len(modes), panel_w=COL_WIDTH * 1.05,
                              panel_h=2.55, sharey=True)

    width = 0.36
    x = np.arange(len(generators))
    for ax, mode in zip(axes, modes):
        sub = df[df["failure_mode"] == mode].set_index("generator")
        std = [sub.loc[g, "std_cot_rate"] if g in sub.index else 0.0 for g in generators]
        narr = [sub.loc[g, "narrative_cot_rate"] if g in sub.index else 0.0 for g in generators]
        b1 = ax.bar(x - width / 2, std, width,
                    color=CONDITION_COLORS["standard_cot"],
                    edgecolor="white", linewidth=0.4,
                    label=CONDITION_LABELS["standard_cot"])
        b2 = ax.bar(x + width / 2, narr, width,
                    color=CONDITION_COLORS["narrative_cot"],
                    edgecolor="white", linewidth=0.4,
                    label=CONDITION_LABELS["narrative_cot"])
        annotate_bars(ax, b1, std, fmt="{:.0f}%", dy=0.013, fs=6.8)
        annotate_bars(ax, b2, narr, fmt="{:.0f}%", dy=0.013, fs=6.8)
        # Annotate the percentage-point drop above each generator pair.
        for xi, s, n in zip(x, std, narr):
            drop = (s - n) * 100
            if drop > 3:
                ax.text(xi, max(s, n) + 0.075,
                        f"−{drop:.0f}pp",
                        ha="center", va="bottom",
                        fontsize=6.5, color=CONDITION_COLORS["narrative_cot"],
                        fontweight="semibold")
        ax.set_title(mode.title(), fontweight="semibold")
        percent_axis(ax, ymax=0.92)
        style_generator_axis(ax, generators)
        ax.set_ylabel("firing rate" if ax is axes[0] else "")
        ax.spines["bottom"].set_position(("outward", 2))

    axes[0].legend(loc="upper right", frameon=False, ncol=1,
                   handlelength=1.4, handletextpad=0.5)
    fig.suptitle("Both failure modes drop sharply under N-CoT on every generator",
                 fontweight="semibold")
    save(fig, OUT / "failure_mode_firing_quartet.pdf")


# ---------------------------------------------------------------------------
# Tier-1 effect sizes with bootstrap CIs (Section 4)
# ---------------------------------------------------------------------------

def fig_tier1_effects() -> None:
    src = OUT / "tier1_effect_sizes_quartet.csv"
    if not src.exists():
        print(f"skip: missing {src}")
        return
    df = pd.read_csv(src)
    df = df[df["generator"] != "gpt-4o"].copy()
    key_vars = ["stakeholder_count", "uncertainty_score"]
    df = df[df["variable"].isin(key_vars)].copy()
    if df.empty:
        return

    generators = [g for g in GENERATOR_ORDER if g in df["generator"].unique().tolist()]
    fig, axes, _ = panel_grid(2, panel_w=COL_WIDTH * 1.05, panel_h=2.7,
                              sharey=True)

    width = 0.36
    x = np.arange(len(generators))
    titles = {"stakeholder_count": "Stakeholder count",
              "uncertainty_score": "Uncertainty score"}
    raw_colour = CONDITION_COLORS["standard_cot"]
    resid_colour = CONDITION_COLORS["narrative_cot"]

    for ax, var in zip(axes, key_vars):
        sub = df[df["variable"] == var].set_index("generator")
        raw = [sub.loc[g, "delta_narr_vs_std"] if g in sub.index else np.nan
               for g in generators]
        raw_lo = [sub.loc[g, "ci_lo"] if g in sub.index else np.nan for g in generators]
        raw_hi = [sub.loc[g, "ci_hi"] if g in sub.index else np.nan for g in generators]
        resid = [sub.loc[g, "resid_delta"] if g in sub.index else np.nan for g in generators]
        resid_lo = [sub.loc[g, "resid_ci_lo"] if g in sub.index else np.nan for g in generators]
        resid_hi = [sub.loc[g, "resid_ci_hi"] if g in sub.index else np.nan for g in generators]

        b1 = ax.bar(x - width / 2, raw, width, color=raw_colour,
                    edgecolor="white", linewidth=0.4, label="raw $\\delta$")
        ax.errorbar(x - width / 2, raw,
                    yerr=[np.subtract(raw, raw_lo), np.subtract(raw_hi, raw)],
                    fmt="none", ecolor="black", capsize=2.5, linewidth=0.8)
        b2 = ax.bar(x + width / 2, resid, width, color=resid_colour,
                    edgecolor="white", linewidth=0.4,
                    label="length-residualised $\\delta$")
        ax.errorbar(x + width / 2, resid,
                    yerr=[np.subtract(resid, resid_lo), np.subtract(resid_hi, resid)],
                    fmt="none", ecolor="black", capsize=2.5, linewidth=0.8)

        ax.axhline(0, color="black", linewidth=0.5)
        for thr, lbl in [(DELTA_MEDIUM, "large"),
                         (DELTA_SMALL, "med."),
                         (DELTA_NEGLIGIBLE, "small")]:
            ax.axhline(thr, color="#999999", linewidth=0.4, linestyle=":")
        ax.text(len(generators) - 0.45, DELTA_MEDIUM + 0.015, "large",
                fontsize=6.0, color="#666666", ha="right")
        ax.set_title(titles[var], fontweight="semibold")
        ax.set_ylim(-1.0, 1.05)
        style_generator_axis(ax, generators)
        if ax is axes[0]:
            ax.set_ylabel(r"Cliff's $\delta$  (N-CoT vs. standard CoT)")

    axes[0].legend(loc="lower left", frameon=False, handlelength=1.4,
                   handletextpad=0.5, ncol=1)
    fig.suptitle("OpenAI / xAI retain large length-residualised effects; "
                 "Anthropic shifts are length-mediated",
                 fontweight="semibold")
    for ax in axes:
        if ax.get_ylabel():
            ax.set_ylabel(r"Cliff's $\delta$  (N-CoT vs. standard CoT)")
    save(fig, OUT / "tier1_effect_sizes_quartet.pdf")


# ---------------------------------------------------------------------------
# Sub-instruction ablation (Section 4.1)
# ---------------------------------------------------------------------------

def fig_subinstruction() -> None:
    src = OUT / "subinstruction_attribution.csv"
    if not src.exists():
        print(f"skip: missing {src}")
        return
    df = pd.read_csv(src)
    section_order = [
        "drop_protagonist", "drop_stakeholders", "drop_consequences",
        "drop_uncertainty", "drop_commitment",
    ]
    var_order = ["stakeholder_count", "max_causal_hops",
                 "uncertainty_score", "n_frameworks"]
    var_labels = {
        "stakeholder_count": "stakeholder\ncount",
        "max_causal_hops":   "causal-hop\ndepth",
        "uncertainty_score": "uncertainty\nscore",
        "n_frameworks":      "framework\ncount",
    }
    sec_labels = {
        "drop_protagonist":  "Drop\nProtagonist",
        "drop_stakeholders": "Drop\nStakeholders",
        "drop_consequences": "Drop\nConsequences",
        "drop_uncertainty":  "Drop\nUncertainty",
        "drop_commitment":   "Drop\nCommitment",
    }
    matrix = np.full((len(section_order), len(var_order)), np.nan)
    for i, sec in enumerate(section_order):
        for j, var in enumerate(var_order):
            row = df[(df["condition"] == sec) & (df["variable"] == var)]
            if not row.empty:
                matrix[i, j] = float(row["delta_drop_vs_full"].values[0])

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(COL_WIDTH * 1.4, 2.6), constrained_layout=True)
    vmax = max(0.5, np.nanmax(np.abs(matrix)))
    # BrBG_r reads as brown (negative = section was carrying the metric) to
    # teal (positive = dropping helped), aligning with the condition palette.
    cmap = matplotlib.colormaps["BrBG_r"]
    im = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(var_order)))
    ax.set_xticklabels([var_labels[v] for v in var_order], fontsize=7.5)
    ax.set_yticks(range(len(section_order)))
    ax.set_yticklabels([sec_labels[s] for s in section_order], fontsize=7.5)
    ax.tick_params(axis="both", length=0)
    ax.grid(False)
    # Identify the most-negative cell per column (i.e. which section, when
    # dropped, hurt this metric the most). Used to bold the diagonal pattern.
    col_argmin = np.nanargmin(matrix, axis=0)
    for i in range(len(section_order)):
        for j in range(len(var_order)):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            colour = "white" if abs(v) > 0.55 else "#222222"
            is_col_min = (i == col_argmin[j]) and v < -0.15
            weight = "bold" if is_col_min else "normal"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=7.5, color=colour, fontweight=weight)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
    cbar.set_label(r"Cliff's $\delta$  (drop vs. full N-CoT)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.set_title("Each substantive section carries its target metric "
                 "(claude-sonnet-4-6, N=3, 30 scenarios)",
                 fontweight="semibold")
    save(fig, OUT / "subinstruction_attribution.pdf")


# ---------------------------------------------------------------------------
# K_C proxy correlation (Section 6)
# ---------------------------------------------------------------------------

def fig_kc_proxy() -> None:
    src = OUT / "kc_proxy_correlation.csv"
    if not src.exists():
        print(f"skip: missing {src}")
        return
    df = pd.read_csv(src)
    df = df[df["generator"] != "gpt-4o"].copy()
    df["generator"] = pd.Categorical(df["generator"], GENERATOR_ORDER, ordered=True)
    df = df.sort_values("generator")

    fig, ax = single_panel(width=COL_WIDTH, height=2.4)
    gens = df["generator"].astype(str).tolist()
    rhos = df["spearman_rho"].astype(float).tolist()
    ns = df["n"].astype(int).tolist()
    colors = [GENERATOR_COLORS.get(g, "#444444") for g in gens]
    bars = ax.bar(range(len(gens)), rhos, color=colors,
                  edgecolor="white", linewidth=0.4)
    for b, n in zip(bars, ns):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.018,
                f"n={n:,}", ha="center", fontsize=6.8, color="#444444")
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() / 2,
                f"{b.get_height():.2f}", ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")
    style_generator_axis(ax, gens)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(r"Spearman $\rho$")
    ax.set_title(r"$\hat{K}_{\mathrm{gzip}}$ tracks the N-CoT contrast across vendors",
                 fontweight="semibold")
    ax.axhline(0.8, color="#999999", linewidth=0.4, linestyle=":")
    save(fig, OUT / "kc_proxy_correlation.pdf")


# ---------------------------------------------------------------------------
# SycophancyEval rates (Section 7)
# ---------------------------------------------------------------------------

def fig_sycophancyeval() -> None:
    src = OUT / "sycophancyeval_results.csv"
    if not src.exists():
        print(f"skip: missing {src}")
        return
    df = pd.read_csv(src)
    df = df[df["generator"] != "gpt-4o"].copy()
    probe_order = ["opinion_mirroring", "retraction_on_pushback", "false_premise"]
    probe_labels = {
        "opinion_mirroring": "Opinion mirroring",
        "retraction_on_pushback": "Retraction on pushback",
        "false_premise": "False premise",
    }
    generators = [g for g in GENERATOR_ORDER if g in df["generator"].unique().tolist()]
    fig, axes, _ = panel_grid(len(probe_order), panel_w=COL_WIDTH * 0.7,
                              panel_h=2.4, sharey=True)
    width = 0.36
    x = np.arange(len(generators))
    for ax, probe in zip(axes, probe_order):
        d = df[df["probe_type"] == probe]
        std = []
        narr = []
        for g in generators:
            std.append(d[(d["generator"] == g) & (d["condition"] == "standard_cot")]
                       ["sycophantic"].astype(float).mean() if len(d) else 0.0)
            narr.append(d[(d["generator"] == g) & (d["condition"] == "narrative_cot")]
                        ["sycophantic"].astype(float).mean() if len(d) else 0.0)
        std = [0.0 if pd.isna(v) else v for v in std]
        narr = [0.0 if pd.isna(v) else v for v in narr]
        b1 = ax.bar(x - width / 2, std, width,
                    color=CONDITION_COLORS["standard_cot"],
                    edgecolor="white", linewidth=0.4,
                    label=CONDITION_LABELS["standard_cot"])
        b2 = ax.bar(x + width / 2, narr, width,
                    color=CONDITION_COLORS["narrative_cot"],
                    edgecolor="white", linewidth=0.4,
                    label=CONDITION_LABELS["narrative_cot"])
        for b, v in list(zip(b1, std)) + list(zip(b2, narr)):
            if v > 0.01:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.004,
                        f"{v*100:.1f}%", ha="center", fontsize=6.5, color="#222222")
        ax.set_title(probe_labels[probe], fontweight="semibold")
        ax.set_ylim(0, 0.085)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
        style_generator_axis(ax, generators)
        if ax is axes[0]:
            ax.set_ylabel("sycophancy rate")
    axes[0].legend(loc="upper left", frameon=False,
                   handlelength=1.4, handletextpad=0.5)
    fig.suptitle("SycophancyEval is saturated on frontier models "
                 "(22 of 24 cells at floor)", fontweight="semibold")
    save(fig, OUT / "sycophancyeval_rates.pdf")


# ---------------------------------------------------------------------------
# Agentic probe rates (Section 7)
# ---------------------------------------------------------------------------

def fig_agentic_probe() -> None:
    src = OUT / "agentic_probe_results.csv"
    if not src.exists():
        print(f"skip: missing {src}")
        return
    df = pd.read_csv(src)
    df = df[df["generator"] != "gpt-4o"].copy()
    scenario_order = ["blackmail", "corporate_espionage"]
    scenario_labels = {"blackmail": "Blackmail",
                       "corporate_espionage": "Corporate espionage"}
    generators = [g for g in GENERATOR_ORDER if g in df["generator"].unique().tolist()]
    fig, axes, _ = panel_grid(len(scenario_order), panel_w=COL_WIDTH * 1.05,
                              panel_h=2.85, sharey=True)
    width = 0.34
    x = np.arange(len(generators))
    cls_palette = {
        "refuse": "#0E7C7B",      # jewel teal (refusal: safe, matches N-CoT)
        "hedge":  "#E2C68A",      # warm wheat (deliberation truncated)
        "harm":   "#B53A2E",      # alarm red (would be very visible if non-zero)
    }
    for ax, scen in zip(axes, scenario_order):
        d = df[df["scenario_id"] == scen]
        for i, cond in enumerate(["standard_cot", "narrative_cot"]):
            stacks = []
            for g in generators:
                sub = d[(d["generator"] == g) & (d["condition"] == cond)]
                n = max(len(sub), 1)
                refuse = (sub["classification"] == "refuse").sum() / n
                hedge = (sub["classification"] == "hedge").sum() / n
                harm = sub["harmful_action"].astype(float).mean() if len(sub) else 0.0
                stacks.append((refuse, hedge, harm))
            xpos = x + (i - 0.5) * width
            refuse = [s[0] for s in stacks]
            hedge = [s[1] for s in stacks]
            harm = [s[2] for s in stacks]
            ax.bar(xpos, refuse, width, color=cls_palette["refuse"],
                   edgecolor="white", linewidth=0.4)
            ax.bar(xpos, hedge, width, bottom=refuse,
                   color=cls_palette["hedge"], edgecolor="white", linewidth=0.4)
            ax.bar(xpos, harm, width, bottom=np.add(refuse, hedge),
                   color=cls_palette["harm"], edgecolor="white", linewidth=0.4)
            for xx in xpos:
                tag = "S" if cond == "standard_cot" else "N"
                ax.text(xx, 1.04, tag, ha="center", va="bottom",
                        fontsize=6.5, color="#666666")
        ax.set_title(scenario_labels[scen], fontweight="semibold", pad=14)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        style_generator_axis(ax, generators)
        if ax is axes[0]:
            ax.set_ylabel("classification share")
    legend_handles = [
        mpatches.Patch(color=cls_palette["refuse"], label="Refuse"),
        mpatches.Patch(color=cls_palette["hedge"], label="Hedge / truncated"),
        mpatches.Patch(color=cls_palette["harm"], label="Harmful action"),
    ]
    axes[-1].legend(handles=legend_handles, loc="lower right", frameon=False,
                    handlelength=1.4, handletextpad=0.5,
                    bbox_to_anchor=(1.0, 0.02))
    fig.suptitle("Zero harmful actions across the quartet under either scaffold "
                 "(S = Std CoT, N = N-CoT, paired per generator)",
                 fontweight="semibold")
    save(fig, OUT / "agentic_probe_rates.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    fig_failure_mode_firing()
    fig_tier1_effects()
    fig_subinstruction()
    fig_kc_proxy()
    fig_sycophancyeval()
    fig_agentic_probe()
    return 0


if __name__ == "__main__":
    sys.exit(main())
