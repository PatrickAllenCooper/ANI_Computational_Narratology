#!/usr/bin/env python3
"""
slurm/su_estimate.py -- CURC Service Unit (SU) burn calculator.

Standalone, stdlib-only.  Estimates the SU cost of a Slurm allocation on
CU Boulder Research Computing (Alpine), using the ONE published figure we
have and being explicit about everything that is not published.

Accounting model implemented here
---------------------------------
CURC documents SU accrual as:

    SU = (cores x hours x 1.0)  +  (gpus x hours x ACCELERATION_FACTOR)

The only *published* acceleration factor is **108.6 SU per A100-GPU-hour**.
Factors for H200, RTX Pro 6000 and GH200 are NOT published.  This tool
therefore reports a BAND:

    low     = A100 factor (108.6)          <- optimistic floor
    central = 2x the A100 factor           <- planning number
    high    = 3x the A100 factor           <- budget-safety ceiling

and prints a loud warning whenever a non-A100 GRES type is selected.

MIG slices
----------
Whether CURC prorates a MIG slice (billing 2/7 of a GPU for
``h200_2g.35gb``) or bills it at the full-GPU rate is ALSO unverified.
Both figures are always printed; the headline number is the conservative
full-GPU-rate one.

Verify the real weights on the cluster with::

    scontrol show partition ah200 | tr ' ' '\\n' | grep -i TRESBillingWeights
    sacctmgr show qos format=Name,UsageFactor%12

Usage
-----
  python slurm/su_estimate.py --partition ah200 --gres h200_2g.35gb \\
                              --gpus 1 --cores 4 --hours 8 --array-tasks 16
  python slurm/su_estimate.py --plan work_manifest.jsonl --gres h200_2g.35gb
  python slurm/su_estimate.py --list-gres
  python slurm/su_estimate.py --smoke
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Published / assumed accounting constants
# ---------------------------------------------------------------------------

#: The single published CURC figure: SU charged per A100-GPU-hour.
#: (Also independently confirmed via TRESBillingWeights -- see below.)
A100_SU_PER_GPU_HOUR: float = 108.6

#: SU charged per core-hour. The module docstring's generic claim was 1.0;
#: CONFIRMED as 0.5 via `scontrol show partition <p> | grep TRESBillingWeights`
#: on 2026-08-17 for every GPU partition this project targets (aa100, ah200,
#: artxpro6000, al40, ami100, gh200 all report CPU=0.5). If this tool is ever
#: pointed at a CPU-only partition (amilan, blanca-*) that has not been
#: checked, re-verify before trusting this constant there.
SU_PER_CORE_HOUR: float = 0.5

#: Multipliers applied to the A100 factor for GRES types whose real
#: acceleration factor CURC has not published.
UNPUBLISHED_BAND: tuple[float, float, float] = (1.0, 2.0, 3.0)

#: QOS usage factors. CONFIRMED 2026-08-17 via a real preflight.sh run on
#: account ucb736_asc1: `scontrol show partition <p>` prints AllowQos=
#: directly, and it is 'gpu-normal'/'gpu-long'/'gpu-testing' (usage factor
#: 0.10 for gpu-testing, matching `sacctmgr show qos` UsageFactor=0.100000)
#: on every GPU partition this project targets (aa100, ah200, artxpro6000,
#: al40, ami100) -- NOT bare 'normal'/'long'/'testing', which are valid QOS
#: names on this account in general (they appear in `sacctmgr show
#: associations`) but are absent from every GPU partition's AllowQos= list.
#: An earlier revision of this file inferred the bare names from a sibling
#: project's aa100 usage (/Users/pat/code/blanc/hpc/*.sh, 30 real job
#: scripts) that turned out not to carry over to this account -- kept here
#: as recognized alternates (harmless if unused) rather than deleted, in
#: case that project's account has a different AllowQos grant. Trust your
#: own preflight.sh's AllowQos= output over this comment if they ever
#: disagree; that is first-party evidence, this is a record of it.
QOS_USAGE_FACTOR: dict[str, float] = {
    "normal": 1.0,
    "long": 1.0,
    "mem": 1.0,
    "testing": 0.10,
    "gpu-normal": 1.0,
    "gpu-long": 1.0,
    "gpu-testing": 0.10,
    "gh200": 1.0,
}


@dataclass(frozen=True)
class GresSpec:
    """One Slurm GRES type string and what we know about its billing."""

    gres: str
    partition: str
    #: Fraction of a whole physical GPU this GRES represents (MIG slices < 1).
    #: Informational only since su_per_gpu_hour below (when set) is CURC's own
    #: already-MIG-aware rate -- see the note on device_fraction below.
    device_fraction: float
    #: True when the acceleration factor for this device IS published/confirmed.
    factor_published: bool
    #: Approximate usable device memory in GB (planning aid only).
    memory_gb: int
    #: CONFIRMED exact SU/GPU-hour from TRESBillingWeights, when known. This is
    #: CURC's real billed rate -- for MIG slices it is NOT simply
    #: device_fraction x the parent GPU's rate (verified: a100_3g.20gb bills at
    #: 54.3, which is ~1/2 of a100-40gb's 108.6, not the 3/7 device_fraction
    #: would suggest -- CURC has its own per-slice weight, not a formula).
    #: None means unconfirmed; estimate() falls back to UNPUBLISHED_BAND.
    su_per_gpu_hour: Optional[float] = None
    note: str = ""


# Partition / GRES table -- verified CURC facts. Rates confirmed 2026-08-17 via
# `scontrol show partition <p> | grep TRESBillingWeights` on account ucb736_asc1
# (see slurm/preflight.sh's U3 section, or Guidance_Documents/curc_runbook.md).
GRES_TABLE: tuple[GresSpec, ...] = (
    # aa100: 12 nodes x 3 A100 -- ALL THREE rates confirmed, not identical to
    # the single "108.6 published" figure (a100_80gb and the MIG slice differ).
    GresSpec("a100-40gb", "aa100", 1.0, True, 40, 108.6, "confirmed 2026-08-17 (matches the published figure)"),
    GresSpec("a100_80gb", "aa100", 1.0, True, 80, 130.396, "confirmed 2026-08-17 -- NOT 108.6, the 80GB card bills higher"),
    GresSpec("a100_3g.20gb", "aa100", 3.0 / 7.0, True, 20, 54.3, "confirmed 2026-08-17 -- CURC's own MIG rate, not a device_fraction proration"),
    # ah200: 8 nodes x 4 H200 141GB -- confirmed 2026-08-17
    GresSpec("h200", "ah200", 1.0, True, 141, 370.406, "confirmed 2026-08-17"),
    GresSpec("h200_3g.71gb", "ah200", 3.0 / 7.0, True, 71, 185.203, "confirmed 2026-08-17, MIG 3g slice"),
    GresSpec("h200_2g.35gb", "ah200", 2.0 / 7.0, True, 35, 123.468, "confirmed 2026-08-17, MIG 2g slice"),
    # artxpro6000: 8 nodes x 4 RTX Pro 6000 96GB -- confirmed 2026-08-17
    GresSpec("rtx_pro_6000", "artxpro6000", 1.0, True, 96, 260.402, "confirmed 2026-08-17"),
    GresSpec("rtx_pro_6000_2g.48gb", "artxpro6000", 0.5, True, 48, 130.201, "confirmed 2026-08-17, MIG 2g slice"),
    GresSpec("rtx_pro_6000_1g.24gb", "artxpro6000", 0.25, True, 24, 65.1, "confirmed 2026-08-17, MIG 1g slice"),
    # al40: 3 nodes x 3 L40 -- confirmed 2026-08-17
    GresSpec("l40", "al40", 1.0, True, 48, 138.417, "confirmed 2026-08-17"),
    # ami100: 8 nodes x 3 MI100 -- confirmed 2026-08-17
    GresSpec("mi100", "ami100", 1.0, True, 32, 85.507, "confirmed 2026-08-17"),
    # gh200: 2 nodes x 1 GH200 -- confirmed 2026-08-17
    GresSpec("gh200", "gh200", 1.0, True, 96, 368.988, "confirmed 2026-08-17"),
)

GRES_BY_NAME: dict[str, GresSpec] = {g.gres: g for g in GRES_TABLE}

#: Valid QOS per partition. CONFIRMED 2026-08-17 via `scontrol show partition
#: <p>`'s AllowQos= on account ucb736_asc1 -- the 'gpu-*' names lead because
#: they are what AllowQos= actually lists; bare 'normal'/'long'/'testing' are
#: kept as recognized-but-not-currently-valid-here alternates (see the
#: QOS_USAGE_FACTOR comment for why they were briefly the default).
QOS_BY_PARTITION: dict[str, tuple[str, ...]] = {
    "aa100": ("gpu-normal", "gpu-long", "gpu-testing", "normal", "long", "testing"),
    "ami100": ("gpu-normal", "gpu-long", "gpu-testing", "normal", "long", "testing"),
    "al40": ("gpu-normal", "gpu-long", "normal", "long"),
    "ah200": ("gpu-normal", "gpu-long", "normal", "long"),
    "artxpro6000": ("gpu-normal", "gpu-long", "normal", "long"),
    "gh200": ("gh200",),
}

#: Per-user concurrency ceiling (MaxTRESPU), by (qos, gres). CONFIRMED
#: 2026-08-17 via `sacctmgr show qos format=Name,MaxTRESPU` on this account.
#: This bounds how many units of a GRES type this user can hold running
#: SIMULTANEOUSLY under a given QOS, across ALL their own jobs combined --
#: e.g. under gpu-normal, at most 4 whole 'h200' units at once, or 6
#: 'h200_2g.35gb' MIG slices at once (independent per-GRES-type ceilings, not
#: a shared pool). Exceeding it does not fail a submission; Slurm queues the
#: excess with reason QOSMaxGRESPerUser until earlier jobs free up headroom.
#: Size `--array=0-N%C` so C does not exceed the relevant cap when every task
#: requests one GRES unit.
QOS_MAX_GRES_PER_USER: dict[tuple[str, str], int] = {
    ("gpu-normal", "a100-40gb"): 6, ("gpu-long", "a100-40gb"): 3,
    ("gpu-normal", "a100_80gb"): 3, ("gpu-long", "a100_80gb"): 1,
    ("gpu-normal", "h200"): 4, ("gpu-long", "h200"): 2,
    ("gpu-normal", "h200_2g.35gb"): 6, ("gpu-long", "h200_2g.35gb"): 2,
    ("gpu-normal", "h200_3g.71gb"): 4, ("gpu-long", "h200_3g.71gb"): 2,
    ("gpu-normal", "l40"): 3, ("gpu-long", "l40"): 3,
    ("gpu-normal", "mi100"): 5, ("gpu-long", "mi100"): 3,
    ("gpu-normal", "rtx_pro_6000"): 4, ("gpu-long", "rtx_pro_6000"): 2,
    ("gpu-normal", "rtx_pro_6000_1g.24gb"): 7, ("gpu-long", "rtx_pro_6000_1g.24gb"): 2,
    ("gpu-normal", "rtx_pro_6000_2g.48gb"): 4, ("gpu-long", "rtx_pro_6000_2g.48gb"): 2,
    ("gpu-testing", "a100_3g.20gb"): 1, ("gpu-testing", "mi100"): 1,
}

#: Assumed single-stream decode throughput (output tokens/second) used by
#: --plan when the caller does not pass --tok-per-s.  These are ASSUMPTIONS,
#: not measurements.  Measure with a --smoke array task and override.
ASSUMED_TOK_PER_S: dict[str, float] = {
    "vllm-8b-mig": 55.0,     # 8B bf16 on a 2g/3g MIG slice, batch 1
    "vllm-8b-full": 110.0,   # 8B bf16 on a whole H200
    "vllm-32b-tp2": 45.0,    # 32B bf16, tensor-parallel 2
    "vllm-70b-tp4": 28.0,    # 70B bf16, tensor-parallel 4
    "hf-8b-mig": 12.0,       # transformers .generate(), no paged attention
    "hf-8b-full": 22.0,
    "hf-32b-tp2": 6.0,
    "hf-70b-tp4": 3.5,
}

DEFAULT_THROUGHPUT_PROFILE = "vllm-8b-mig"

#: Fraction of ``max_tokens`` a typical completion actually uses.  Used to
#: produce a "typical" alongside the "worst case" plan estimate.
TYPICAL_COMPLETION_FRACTION = 0.55


# ---------------------------------------------------------------------------
# Core estimator
# ---------------------------------------------------------------------------

@dataclass
class Estimate:
    """Result of one SU estimate."""

    gres: str
    partition: str
    qos: str
    gpus: float
    cores: int
    hours: float
    usage_factor: float
    core_su: float
    gpu_su_full_rate: tuple[float, float, float]
    gpu_su_prorated: tuple[float, float, float]
    factor_published: bool
    device_fraction: float

    @property
    def total_full_rate(self) -> tuple[float, float, float]:
        return tuple(self.core_su + g for g in self.gpu_su_full_rate)  # type: ignore[return-value]

    @property
    def total_prorated(self) -> tuple[float, float, float]:
        return tuple(self.core_su + g for g in self.gpu_su_prorated)  # type: ignore[return-value]


def resolve_gres(name: str) -> GresSpec:
    """Look up a GRES type string, tolerating a ``gpu:`` prefix."""
    cleaned = name.strip()
    if cleaned.startswith("gpu:"):
        cleaned = cleaned[len("gpu:"):]
    # Tolerate a trailing ":N" count.
    if ":" in cleaned:
        cleaned = cleaned.split(":", 1)[0]
    if cleaned not in GRES_BY_NAME:
        known = ", ".join(sorted(GRES_BY_NAME))
        raise SystemExit(f"unknown GRES type {name!r}. Known: {known}")
    return GRES_BY_NAME[cleaned]


def estimate(
    *,
    gres: str,
    gpus: float,
    cores: int,
    hours: float,
    partition: Optional[str] = None,
    qos: str = "gpu-normal",
) -> Estimate:
    """Estimate SU burn for one allocation shape.

    Args:
        gres: GRES type string, e.g. ``h200_2g.35gb`` (``gpu:`` prefix ok).
        gpus: number of GRES units requested.
        cores: ``--cpus-per-task`` x ``--ntasks`` for the job.
        hours: wall-clock hours the allocation is held.
        partition: partition name; defaults to the GRES table's partition.
        qos: QOS name, used only for the documented usage factor.

    Returns:
        An :class:`Estimate` carrying both the conservative (full-GPU-rate)
        and the optimistic (MIG-prorated) SU bands.
    """
    spec = resolve_gres(gres)
    part = partition or spec.partition
    usage = QOS_USAGE_FACTOR.get(qos, 1.0)

    core_su = cores * hours * SU_PER_CORE_HOUR * usage

    if spec.su_per_gpu_hour is not None:
        # Exact, confirmed rate: no band, no proration guess. CURC bills MIG
        # slices at their OWN directly-published rate (verified: a100_3g.20gb
        # is 54.3, not 108.6 x 3/7), so "full-GPU-rate" vs "MIG-prorated" is no
        # longer two competing hypotheses for a confirmed GRES type -- there is
        # exactly one real number, and both columns report it identically.
        exact = gpus * hours * spec.su_per_gpu_hour * usage
        gpu_full = (exact, exact, exact)
        gpu_prorated = (exact, exact, exact)
    else:
        # No confirmed rate for this GRES type: fall back to the old
        # A100-relative guess. Kept for any future partition/GRES this table
        # has not yet had a real TRESBillingWeights reading for.
        base = gpus * hours * A100_SU_PER_GPU_HOUR * usage
        gpu_full = tuple(base * m for m in UNPUBLISHED_BAND)
        gpu_prorated = tuple(base * m * spec.device_fraction for m in UNPUBLISHED_BAND)

    return Estimate(
        gres=spec.gres,
        partition=part,
        qos=qos,
        gpus=gpus,
        cores=cores,
        hours=hours,
        usage_factor=usage,
        core_su=core_su,
        gpu_su_full_rate=gpu_full,  # type: ignore[arg-type]
        gpu_su_prorated=gpu_prorated,  # type: ignore[arg-type]
        factor_published=spec.factor_published,
        device_fraction=spec.device_fraction,
    )


# ---------------------------------------------------------------------------
# Plan mode: read a work manifest and estimate the whole sweep
# ---------------------------------------------------------------------------

@dataclass
class PlanShape:
    """Aggregate work implied by a manifest."""

    cells: int
    cached: int
    todo: int
    max_output_tokens: int
    typical_output_tokens: int


def _out_dir(explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit)
    return Path("./divergence_study_outputs")


def _safe(text: str) -> str:
    """Filename sanitiser -- byte-identical to ``scripts.run_phase1_quartet._safe``."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", text)


def _cell_cache_path(cell: dict, out_dir: Path) -> Path:
    """Reproduce the array worker's cache-path rule (see array_generate.sbatch)."""
    explicit = cell.get("cache_path")
    if explicit:
        p = Path(explicit)
        return p if p.is_absolute() else out_dir / p
    kind = cell.get("cache_kind", "bm_gen")
    model = str(cell.get("model", "unknown"))
    arm = str(cell.get("arm", "raw"))
    item = str(cell.get("item_id") or cell.get("problem_id") or cell.get("id") or "0")
    return out_dir / f"{kind}_{_safe(model)}_{arm}_{_safe(item)}.json"


def read_plan(manifest: Path, *, out_dir: Path, default_max_tokens: int) -> PlanShape:
    """Scan a JSONL work manifest and summarise remaining work.

    Cells whose per-cell JSON cache already exists are counted as ``cached``
    and excluded from the token totals, so re-running this after a partial
    sweep reports only what is left to burn.
    """
    cells = cached = 0
    max_tokens_total = 0
    with manifest.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                cell = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{manifest}:{lineno}: bad JSON: {exc}") from exc
            cells += 1
            if _cell_cache_path(cell, out_dir).exists():
                cached += 1
                continue
            max_tokens_total += int(cell.get("max_tokens", default_max_tokens))
    typical = int(round(max_tokens_total * TYPICAL_COMPLETION_FRACTION))
    return PlanShape(
        cells=cells,
        cached=cached,
        todo=cells - cached,
        max_output_tokens=max_tokens_total,
        typical_output_tokens=typical,
    )


def plan_hours(tokens: int, tok_per_s: float) -> float:
    """GPU-slice-hours needed to decode ``tokens`` output tokens."""
    if tok_per_s <= 0:
        raise SystemExit("--tok-per-s must be > 0")
    return tokens / tok_per_s / 3600.0


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

BANNER = "=" * 74


def _fmt_band(band: tuple[float, float, float], published: bool) -> str:
    lo, mid, hi = band
    if published:
        return f"{lo:>12,.0f} SU"
    return f"{lo:>12,.0f} / {mid:,.0f} / {hi:,.0f} SU  (low/central/high)"


def render_estimate(est: Estimate, *, label: str = "ALLOCATION") -> str:
    lines = [
        BANNER,
        f" {label}",
        BANNER,
        f"  partition        : {est.partition}",
        f"  gres             : gpu:{est.gres}:{est.gpus:g}",
        f"  qos              : {est.qos}   (usage factor {est.usage_factor:g})",
        f"  cores            : {est.cores}",
        f"  wall-clock hours : {est.hours:,.2f}",
        "",
        f"  core-hour SU     : {est.core_su:>12,.1f} SU"
        f"   ({est.cores} cores x {est.hours:,.2f} h x {SU_PER_CORE_HOUR:g})",
        f"  GPU SU (full-GPU rate) : {_fmt_band(est.gpu_su_full_rate, est.factor_published)}",
    ]
    if est.device_fraction < 1.0:
        lines.append(
            f"  GPU SU (MIG prorated x{est.device_fraction:.3f}) :"
            f" {_fmt_band(est.gpu_su_prorated, est.factor_published)}"
        )
    lines += [
        "",
        f"  TOTAL (conservative)   : {_fmt_band(est.total_full_rate, est.factor_published)}",
    ]
    if est.device_fraction < 1.0:
        lines.append(
            f"  TOTAL (if MIG prorated): {_fmt_band(est.total_prorated, est.factor_published)}"
        )
    return "\n".join(lines)


def render_warnings(est: Estimate) -> str:
    warn: list[str] = []
    if not est.factor_published:
        warn.append(
            f"UNPUBLISHED BILLING FACTOR: CURC publishes only the A100 figure\n"
            f"    ({A100_SU_PER_GPU_HOUR} SU per A100-GPU-hour).  The acceleration factor for\n"
            f"    '{est.gres}' is NOT published and could plausibly be 2-3x the A100\n"
            f"    factor.  The low number above assumes parity with an A100 and is a\n"
            f"    FLOOR, not a forecast.  Budget against the 'high' column."
        )
    if est.device_fraction < 1.0 and not est.factor_published:
        warn.append(
            f"MIG BILLING UNVERIFIED: '{est.gres}' is a MIG slice"
            f" ({est.device_fraction:.3f} of a device).\n"
            "    Whether Slurm bills a slice prorated or at the full-GPU rate depends on\n"
            "    this cluster's TRESBillingWeights, which we have not read.  The headline\n"
            "    TOTAL assumes the FULL-GPU rate (worst case)."
        )
    if est.qos in ("gpu-testing", "testing"):
        warn.append(
            f"QOS {est.qos!r} is valid ONLY on partitions aa100 and ami100, is capped at\n"
            "    1 hour and 5 concurrent jobs, and is documented as billed at 10%."
        )
    valid = QOS_BY_PARTITION.get(est.partition)
    if valid and est.qos not in valid:
        warn.append(
            f"QOS '{est.qos}' is NOT valid on partition '{est.partition}'."
            f"  Valid: {', '.join(valid)}."
        )
    cap = QOS_MAX_GRES_PER_USER.get((est.qos, est.gres))
    if cap is not None and est.gpus > cap:
        warn.append(
            f"THIS JOB ALONE ({est.gpus:g}x {est.gres!r}) EXCEEDS the per-user cap "
            f"of {cap} under QOS {est.qos!r} (MaxTRESPU) -- it cannot run as shaped, "
            f"regardless of allocation balance. Lower --gpus, request a different QOS "
            f"(e.g. gpu-normal usually allows more concurrent units than gpu-long -- "
            f"see QOS_MAX_GRES_PER_USER), or split it into multiple smaller jobs."
        )
    elif cap is not None:
        note = (
            f"per-user cap under QOS {est.qos!r} for {est.gres!r}: {cap} concurrent "
            f"unit(s) (MaxTRESPU) across ALL your running jobs. If this is one task of "
            f"an sbatch --array=..%N sweep, keep N x {est.gpus:g} <= {cap} or the "
            f"excess tasks will queue instead of running."
        )
        warn.append(note)
    if not warn:
        return ""
    body = "\n".join(f"  [!] {w}" for w in warn)
    return f"\n{BANNER}\n WARNINGS\n{BANNER}\n{body}"


def render_verify_hint() -> str:
    return (
        f"\n{BANNER}\n"
        " HOW TO REPLACE THESE GUESSES WITH FACTS (run on a CURC login node)\n"
        f"{BANNER}\n"
        "  scontrol show partition ah200 | tr ' ' '\\n' | grep -i TRESBillingWeights\n"
        "  sacctmgr show qos format=Name,UsageFactor%12,MaxWall,MaxSubmitJobsPU\n"
        "  sreport cluster AccountUtilizationByUser start=$(date -d '-7 days' +%F) -t hours\n"
        "  # then hard-code the observed weights at the top of this file.\n"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _list_gres() -> int:
    print(f"{'GRES type':<24} {'partition':<14} {'frac':>6} {'mem GB':>7}  note")
    print("-" * 88)
    for spec in GRES_TABLE:
        print(
            f"{spec.gres:<24} {spec.partition:<14} {spec.device_fraction:>6.3f}"
            f" {spec.memory_gb:>7}  {spec.note}"
        )
    print()
    print("QOS by partition:")
    for part, qoses in QOS_BY_PARTITION.items():
        print(f"  {part:<14} {', '.join(qoses)}")
    return 0


def _run_plan(args: argparse.Namespace) -> int:
    manifest = Path(args.plan)
    if not manifest.exists():
        raise SystemExit(f"manifest not found: {manifest}")
    out_dir = _out_dir(args.out_dir)
    shape = read_plan(manifest, out_dir=out_dir, default_max_tokens=args.default_max_tokens)

    if args.tok_per_s is not None:
        tok_per_s = args.tok_per_s
        profile = "user-supplied"
    else:
        profile = args.throughput_profile
        if profile not in ASSUMED_TOK_PER_S:
            raise SystemExit(
                f"unknown --throughput-profile {profile!r}; "
                f"known: {', '.join(sorted(ASSUMED_TOK_PER_S))}"
            )
        tok_per_s = ASSUMED_TOK_PER_S[profile]

    print(BANNER)
    print(" SWEEP PLAN")
    print(BANNER)
    print(f"  manifest            : {manifest}")
    print(f"  cache dir           : {out_dir}")
    print(f"  cells in manifest   : {shape.cells:,}")
    print(f"  already cached      : {shape.cached:,}  (skipped, cost nothing)")
    print(f"  cells still to run  : {shape.todo:,}")
    print(f"  output tokens (max) : {shape.max_output_tokens:,}")
    print(
        f"  output tokens (typ) : {shape.typical_output_tokens:,}"
        f"   ({TYPICAL_COMPLETION_FRACTION:.0%} of max_tokens)"
    )
    print(f"  throughput assumed  : {tok_per_s:g} tok/s   (profile: {profile})")

    for label, tokens in (
        ("TYPICAL (completions use %.0f%% of max_tokens)" % (TYPICAL_COMPLETION_FRACTION * 100), shape.typical_output_tokens),
        ("WORST CASE (every completion hits max_tokens)", shape.max_output_tokens),
    ):
        hours = plan_hours(tokens, tok_per_s)
        hours *= 1.0 + args.overhead
        est = estimate(
            gres=args.gres,
            gpus=args.gpus,
            cores=args.cores,
            hours=hours,
            partition=args.partition,
            qos=args.qos,
        )
        print()
        print(render_estimate(est, label=label))
        wall = hours / max(args.array_tasks, 1)
        print(
            f"  GPU-slice-hours  : {hours:,.2f}"
            f"   (incl. {args.overhead:.0%} startup/model-load overhead)"
        )
        print(
            f"  wall-clock       : {wall:,.2f} h spread over {args.array_tasks} array task(s)"
        )
        if wall > args.task_time_limit:
            need = int(-(-wall // args.task_time_limit))  # ceil
            print(
                f"  [!] one task would need {wall:,.2f} h > --task-time-limit"
                f" {args.task_time_limit:g} h;"
                f" use at least {args.array_tasks * need} array tasks,"
                " or rely on cache-skip resume across resubmissions."
            )

    est_for_warnings = estimate(
        gres=args.gres, gpus=args.gpus, cores=args.cores, hours=1.0,
        partition=args.partition, qos=args.qos,
    )
    warn = render_warnings(est_for_warnings)
    if warn:
        print(warn)
    print(render_verify_hint())
    return 0


def _run_single(args: argparse.Namespace) -> int:
    est = estimate(
        gres=args.gres,
        gpus=args.gpus,
        cores=args.cores,
        hours=args.hours,
        partition=args.partition,
        qos=args.qos,
    )
    print(render_estimate(est))
    if args.array_tasks > 1:
        total = est.total_full_rate
        print()
        print(f"  x {args.array_tasks} array tasks -> "
              f"{_fmt_band(tuple(v * args.array_tasks for v in total), est.factor_published)}")  # type: ignore[arg-type]
    warn = render_warnings(est)
    if warn:
        print(warn)
    if not est.factor_published:
        # Nothing left to verify for a confirmed GRES type -- printing this
        # unconditionally used to tell the user to go re-derive a rate this
        # file already has hardcoded.
        print(render_verify_hint())
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Estimate CURC SU burn for a GPU allocation or a planned sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Only the A100 acceleration factor (108.6 SU/GPU-hour) is published.\n"
            "H200 / RTX Pro 6000 / GH200 factors are estimated as a 1x-3x band."
        ),
    )
    ap.add_argument("--partition", default=None,
                    help="Slurm partition (default: inferred from --gres)")
    ap.add_argument("--gres", default="h200_2g.35gb",
                    help="GRES type string, e.g. h200_2g.35gb (default: %(default)s)")
    ap.add_argument("--gpus", type=float, default=1.0,
                    help="GRES units requested (default: %(default)s)")
    ap.add_argument("--cores", type=int, default=4,
                    help="total CPU cores held by the job (default: %(default)s)")
    ap.add_argument("--hours", type=float, default=8.0,
                    help="wall-clock hours per task (default: %(default)s)")
    ap.add_argument("--qos", default="gpu-normal",
                    help="QOS name (default: %(default)s; CONFIRMED 2026-08-17 "
                         "via AllowQos= on aa100/ah200/artxpro6000/al40/ami100 "
                         "-- see QOS_USAGE_FACTOR)")
    ap.add_argument("--array-tasks", type=int, default=1,
                    help="number of array tasks in the sweep (default: %(default)s)")

    ap.add_argument("--plan", default=None, metavar="MANIFEST.jsonl",
                    help="estimate total burn for a work manifest instead of a fixed shape")
    ap.add_argument("--out-dir", default=None,
                    help="per-cell cache dir used to discount already-done cells "
                         "(default: ./divergence_study_outputs)")
    ap.add_argument("--tok-per-s", type=float, default=None,
                    help="measured output tokens/sec; overrides --throughput-profile")
    ap.add_argument("--throughput-profile", default=DEFAULT_THROUGHPUT_PROFILE,
                    choices=sorted(ASSUMED_TOK_PER_S),
                    help="assumed decode throughput (default: %(default)s)")
    ap.add_argument("--default-max-tokens", type=int, default=2048,
                    help="max_tokens assumed for manifest cells that omit it")
    ap.add_argument("--overhead", type=float, default=0.15,
                    help="fractional startup/model-load overhead added to GPU hours "
                         "(default: %(default)s)")
    ap.add_argument("--task-time-limit", type=float, default=8.0,
                    help="per-task wall limit in hours, used for the sharding hint "
                         "(default: %(default)s)")

    ap.add_argument("--list-gres", action="store_true",
                    help="print the GRES/partition/QOS table and exit")
    ap.add_argument("--smoke", action="store_true",
                    help="run a tiny self-check with a synthetic manifest and exit")
    return ap


def _smoke() -> int:
    import tempfile

    print("[smoke] GRES table entries:", len(GRES_TABLE))
    # 'gpu-normal' -- CONFIRMED 2026-08-17 via a real preflight.sh AllowQos=
    # reading on this account. This is now the default; the assert also
    # doubles as a regression test that the discount table is wired up.
    est = estimate(gres="a100-40gb", gpus=1, cores=4, hours=1.0, qos="gpu-normal")
    expected = 4 * SU_PER_CORE_HOUR + 1 * A100_SU_PER_GPU_HOUR
    assert abs(est.total_full_rate[0] - expected) < 1e-6, est.total_full_rate
    print(f"[smoke] 1 A100-hour + 4 cores = {est.total_full_rate[0]:.1f} SU (expected {expected:.1f})")

    test_est = estimate(gres="a100-40gb", gpus=1, cores=1, hours=1.0, qos="gpu-testing")
    assert abs(test_est.total_full_rate[0] - (SU_PER_CORE_HOUR + A100_SU_PER_GPU_HOUR) * 0.10) < 1e-6
    print(f"[smoke] same under gpu-testing (10%) = {test_est.total_full_rate[0]:.2f} SU")

    # Bare 'normal'/'testing' spellings must still resolve identically -- they
    # are valid QOS names on this account in general (sacctmgr show
    # associations), just not on any GPU partition's AllowQos= (an earlier
    # revision of this file briefly had these as the default; see the
    # QOS_USAGE_FACTOR comment for why).
    bare_est = estimate(gres="a100-40gb", gpus=1, cores=4, hours=1.0, qos="normal")
    assert bare_est.total_full_rate == est.total_full_rate
    bare_test_est = estimate(gres="a100-40gb", gpus=1, cores=1, hours=1.0, qos="testing")
    assert bare_test_est.total_full_rate == test_est.total_full_rate
    print("[smoke] bare 'normal'/'testing' spellings still resolve identically")

    # MaxTRESPU warning fires when a single job's request alone exceeds the
    # confirmed per-user concurrency cap.
    over_cap = estimate(gres="h200", gpus=6, cores=32, hours=1.0, qos="gpu-normal")
    warning_text = render_warnings(over_cap)
    assert "EXCEEDS the per-user cap" in warning_text, warning_text
    print("[smoke] MaxTRESPU over-cap warning fires correctly")

    mig = estimate(gres="h200_2g.35gb", gpus=1, cores=4, hours=8.0)
    print(f"[smoke] 8h on h200_2g.35gb -> confirmed exact rate "
          f"{mig.total_full_rate[0]:,.0f} SU")
    # Confirmed 2026-08-17 (was previously the unpublished-band example).
    assert mig.factor_published
    assert mig.total_full_rate[0] == mig.total_full_rate[1] == mig.total_full_rate[2]

    with tempfile.TemporaryDirectory() as td:
        man = Path(td) / "smoke_manifest.jsonl"
        cells = [
            {"cache_kind": "bm_gen", "model": "Qwen3-8B", "arm": "raw",
             "item_id": f"p{i}", "max_tokens": 2048, "system": "", "user": "x"}
            for i in range(120)
        ]
        man.write_text("\n".join(json.dumps(c) for c in cells) + "\n", encoding="utf-8")
        shape = read_plan(man, out_dir=Path(td) / "cache", default_max_tokens=2048)
        assert shape.cells == 120 and shape.todo == 120, shape
        print(f"[smoke] manifest scan: {shape.cells} cells, "
              f"{shape.max_output_tokens:,} worst-case output tokens")
        hours = plan_hours(shape.typical_output_tokens, ASSUMED_TOK_PER_S["vllm-8b-mig"])
        print(f"[smoke] typical GPU-slice-hours at "
              f"{ASSUMED_TOK_PER_S['vllm-8b-mig']:g} tok/s = {hours:.2f} h")
    print("[smoke] OK")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        return _smoke()
    if args.list_gres:
        return _list_gres()
    if args.plan:
        return _run_plan(args)
    return _run_single(args)


if __name__ == "__main__":
    sys.exit(main())
