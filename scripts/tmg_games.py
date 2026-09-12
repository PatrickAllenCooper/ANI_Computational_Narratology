"""
scripts/tmg_games.py -- The binding joint-plan instrument on the 144 TMGBench
"classic" ordinal 2x2 games (Robinson-Goforth topology): data loader,
first-principles gold that must reproduce the canonical gold file, leak-
controlled item rendering, and the PLAN_xx forced-verdict instrument.

This module makes NO model calls and spends nothing. It is the shared data
layer for the deliberation runner (three seats: row-player advocate,
column-player advocate, and a third seat that is either the naive neutral
reader or the planner/mediator), for the single-agent runner and its
``--masked`` hypothesis-only baseline, and for the analyzer (per-gold_type
accuracy, Nash-pull, grip G1/G2/G3).

DATA
----
``data/tmgbench_classic/NNN.txt`` -- 144 files, one per game, each a
``[Scenario]`` block with a payoff table

    | A \\ B | B1    | B2    |
    | A1    | a \\ b | a \\ b |
    | A2    | a \\ b | a \\ b |

where ``a`` is the row player's payoff and ``b`` the column player's, each
an ordinal rank 1..4 (larger better). Every player's four payoffs are a
permutation of {1,2,3,4}: there are no ties, so weak and strict pure Nash
coincide and weak-with-one-strict dominance equals strict dominance.

``data/tmgbench_classic_gold.json`` -- the CANONICAL gold record per game id
(``nash``, ``strict_nash``, ``domA``, ``domB``, ``pareto``, ``welfare``,
``both_best``, ``constant_sum``, ``nash_pareto``, ``nash_welfare``, ``w``,
``type``, ``pA``, ``pB``, ``gold_type``, ``gold_cell``). Both files are
sha256-pinned below (GAMES_SHA256, GOLD_SHA256); ``load_games`` refuses to
load if either pin fails. The code REPRODUCES the gold; it never overrides it.

DERIVED GOLD RULES (first principles; the selftest requires 144/144 agreement
with the canonical file on gold_type AND gold_cell, and on every other field)
-------------------------------------------------------------------------------
Let N = pure Nash cells (weak best responses), P = Pareto-efficient cells
(not weakly dominated with at least one strict inequality), and for N
non-empty D = {c in P : c is STRICTLY better for BOTH players than EVERY
Nash cell}.

  1. constant_sum (a + b == 5 in every cell)        -> ZERO_SUM, no cell
  2. |P| == 1, p = the one efficient cell:
       |N| == 1 and N == {p}                         -> HARMONY, p
       |N| >= 2 and p in N                           -> ASSURANCE, p
       |N| == 0                                      -> CYCLIC_UNIQUE_IDEAL, p
       p not in N                                    -> DILEMMA_UNIQUE_IDEAL, p
  3. |P| > 1 and N non-empty:
       |D| == 1 and |N| == 1                         -> DILEMMA_BOTH_PREFER, d
       |D| == 1 and |N| >= 2                         -> ASSURANCE_LIKE_MULTI, d
       |D| >  1                                      -> AMBIGUOUS_IDEAL_NEEDS_CARDINAL, no cell
       |D| == 0 and |N| == 1                         -> INCENTIVE_COMPATIBLE_EFFICIENT, n
       |D| == 0 and |N| >= 2                         -> CONFLICT_OF_EQUILIBRIA, no cell
  4. |P| > 1 and N empty                             -> CYCLIC_NO_IDEAL, no cell

On strict-ordinal 2x2 games a unique efficient cell must be the (4,4) cell
(dominance is transitive, so the sole undominated cell weakly dominates every
other cell in both coordinates, hence holds both maxima), and a (4,4) cell is
always a Nash cell; the two rule-2 sub-branches without p in N are therefore
unreachable, as are the two rule-3 branches with |D| > 1 or |N| >= 2 with
|D| == 1. They are kept so the rule set is complete and auditable; the
selftest pins that exactly seven types occur, with the registered counts:
INCENTIVE_COMPATIBLE_EFFICIENT 70, HARMONY 27, CYCLIC_NO_IDEAL 16,
CONFLICT_OF_EQUILIBRIA 9, ASSURANCE 9, DILEMMA_BOTH_PREFER 7, ZERO_SUM 6.

REGIME AND VERDICT
------------------
Binding joint plan: before playing, the two players may sign a binding
agreement on a joint plan (one option each). The question put to every seat
is which joint plan both should accept, if exactly one such plan exists.
Verdict tokens: PLAN_11, PLAN_12, PLAN_21, PLAN_22 (row player's DISPLAYED
option number, then column player's DISPLAYED option number) and
NO_UNIQUE_PLAN. Gold = gold_cell mapped through the item's display
permutation; gold = NO_UNIQUE_PLAN when gold_cell is null (the 31 cyclic /
conflict / zero-sum games).

Extent (registered): value region = DILEMMA_BOTH_PREFER + ASSURANCE (16
games), where a mediating third seat can add value; neutral region =
INCENTIVE_COMPATIBLE_EFFICIENT + HARMONY (97), where incentives already
align; cannot_assist = CYCLIC + CONFLICT + ZERO_SUM (31), where no unique
plan exists.

LEAK CONTROLS
-------------
In canonical orientation the gold cell of every DILEMMA / ASSURANCE / HARMONY
game sits at the same position, [0,1] (the selftest pins this), so an item
rendered as-is would let a model score by position alone. Per item, seeded by
(game_id, sample_idx):
  (a) the row player's two options and the column player's two options are
      independently permuted and relabelled 1/2 in the permuted order;
  (b) player names are drawn from a fixed list of valence-neutral names with
      distinct initials (option labels are initial + 1/2); the words
      cooperate / defect / cooperation / defection / prisoner / dilemma /
      betray / trust never appear in any rendered text (selftest-enforced,
      case-insensitive substring);
  (c) the payoff table is presented in the permuted orientation;
  (d) ``masked=True`` replaces every payoff number with "?" so the text
      carries only surface form -- the hypothesis-only baseline, whose
      accuracy must sit at the base-rate floor or the item text leaks
      (``masked_floor_report`` tests exactly that with an item-clustered
      bootstrap).
Masked and unmasked views of the same (game_id, sample_idx) share the same
permutation, names and gold, so the masked baseline is item-matched.

VERDICT PARSING
---------------
The instrument is registered with scripts.verdict_format via
``register_instrument`` -- the extension path that file provides and the one
run_crowdgold_dilemma.py uses -- so verdict_format.py itself is not edited.
``parse_plan_verdict`` normalises only spacing/punctuation variants of the
exact tokens ("PLAN 12", "PLAN-12", "PLAN_1_2", "NO UNIQUE PLAN") and then
applies verdict_format's conservative last-line extractor. Anything else,
including a stray UNRESOLVED (never offered by GAME_VERDICT_INSTRUCTION), a
single-digit "PLAN_1" or the zero-separator "PLAN12" (a near-miss, not a
spacing variant), is NOVERDICT. A verdict line that names MORE THAN ONE
distinct legal token ("VERDICT: PLAN_12 or PLAN_21", "VERDICT: PLAN_12;
NO_UNIQUE_PLAN") is a hedge, not a verdict, and is NOVERDICT too: on this
instrument two acceptable plans is exactly what NO_UNIQUE_PLAN exists for,
so reading the first token would score the hedge as a wrong committal PLAN
and inflate Nash-pull. The cost is that a contrastive committal
("VERDICT: PLAN_21, since PLAN_12 leaves R at 1") is also NOVERDICT and
counts against the 5% guard -- the conservative side of the registered
"never coerce" rule.

Usage:
  python -m scripts.tmg_games --selftest
  python -m scripts.tmg_games --summary
  python -m scripts.tmg_games --show 111 --sample-idx 3 [--masked]
  python -m scripts.tmg_games --extract "VERDICT: PLAN 12"
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import sys
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from scripts.verdict_format import (
    INSTRUMENTS,
    NOVERDICT,
    UNRESOLVED,
    extract_verdict_detail,
    register_instrument,
    score_against_gold,
    verdict_instruction,
)

# ---------------------------------------------------------------------------
# Data locations and byte pins
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "tmgbench_classic"
GOLD_PATH = REPO_ROOT / "data" / "tmgbench_classic_gold.json"
#: Same directory run_phase1_quartet.OUT_DIR names (relative to the repo
#: root); anchored on __file__ here so importing this module has no
#: mkdir/dotenv side effects.
OUT_DIR = REPO_ROOT / "divergence_study_outputs"

N_GAMES = 144
#: sha256 over sorted (basename \\0 bytes \\0) of the 144 game files.
GAMES_SHA256 = "53abf23320da1aa0a557a7aad3f2bfcffc817410f915912f6f7bd62b384c2345"
#: sha256 of data/tmgbench_classic_gold.json, byte for byte.
GOLD_SHA256 = "65509ea563665d91edf28c953ca8df293e16af9ce230d7548e3964c02fcb6a3b"

Cell = tuple[int, int]
Matrix = tuple[tuple[int, int], tuple[int, int]]
CELLS: tuple[Cell, ...] = ((0, 0), (0, 1), (1, 0), (1, 1))

# ---------------------------------------------------------------------------
# Gold types, regions, registered counts
# ---------------------------------------------------------------------------

ZERO_SUM = "ZERO_SUM"
HARMONY = "HARMONY"
ASSURANCE = "ASSURANCE"
DILEMMA_BOTH_PREFER = "DILEMMA_BOTH_PREFER"
INCENTIVE_COMPATIBLE_EFFICIENT = "INCENTIVE_COMPATIBLE_EFFICIENT"
CONFLICT_OF_EQUILIBRIA = "CONFLICT_OF_EQUILIBRIA"
CYCLIC_NO_IDEAL = "CYCLIC_NO_IDEAL"
# Rule branches that are unreachable on strict-ordinal 2x2 games (see the
# module docstring). Kept so the rule set is complete; the selftest pins that
# none of them occurs.
CYCLIC_UNIQUE_IDEAL = "CYCLIC_UNIQUE_IDEAL"
DILEMMA_UNIQUE_IDEAL = "DILEMMA_UNIQUE_IDEAL"
ASSURANCE_LIKE_MULTI = "ASSURANCE_LIKE_MULTI"
AMBIGUOUS_IDEAL_NEEDS_CARDINAL = "AMBIGUOUS_IDEAL_NEEDS_CARDINAL"

REGION_VALUE = "value"
REGION_NEUTRAL = "neutral"
REGION_CANNOT_ASSIST = "cannot_assist"
REGIONS = (REGION_VALUE, REGION_NEUTRAL, REGION_CANNOT_ASSIST)

REGION_OF: dict[str, str] = {
    DILEMMA_BOTH_PREFER: REGION_VALUE,
    ASSURANCE: REGION_VALUE,
    INCENTIVE_COMPATIBLE_EFFICIENT: REGION_NEUTRAL,
    HARMONY: REGION_NEUTRAL,
    CYCLIC_NO_IDEAL: REGION_CANNOT_ASSIST,
    CONFLICT_OF_EQUILIBRIA: REGION_CANNOT_ASSIST,
    ZERO_SUM: REGION_CANNOT_ASSIST,
}
GOLD_TYPES = tuple(REGION_OF)

#: The registered per-type counts on the 144 classic games.
REGISTERED_TYPE_COUNTS: dict[str, int] = {
    INCENTIVE_COMPATIBLE_EFFICIENT: 70, HARMONY: 27, CYCLIC_NO_IDEAL: 16,
    CONFLICT_OF_EQUILIBRIA: 9, ASSURANCE: 9, DILEMMA_BOTH_PREFER: 7,
    ZERO_SUM: 6,
}
REGISTERED_REGION_COUNTS: dict[str, int] = {
    REGION_VALUE: 16, REGION_NEUTRAL: 97, REGION_CANNOT_ASSIST: 31,
}

# ---------------------------------------------------------------------------
# Verdict instrument (registered with verdict_format, not edited into it)
# ---------------------------------------------------------------------------

GAME_INSTRUMENT = "tmg_plan_verdict"
PLAN_11, PLAN_12, PLAN_21, PLAN_22 = "PLAN_11", "PLAN_12", "PLAN_21", "PLAN_22"
NO_UNIQUE_PLAN = "NO_UNIQUE_PLAN"
PLAN_VERDICTS = (PLAN_11, PLAN_12, PLAN_21, PLAN_22)
VERDICT_SET = PLAN_VERDICTS + (NO_UNIQUE_PLAN,)

#: The forced final line, verbatim. Pinned as a literal (and re-derived from
#: verdict_format in the selftest) so the wording is a single auditable
#: string. UNRESOLVED is deliberately NOT offered: NO_UNIQUE_PLAN is this
#: instrument's own legal "there is no answer" token, and it is a committal
#: verdict that can be right or wrong.
GAME_VERDICT_INSTRUCTION = (
    "End your response with exactly one line: VERDICT: PLAN_11, "
    "VERDICT: PLAN_12, VERDICT: PLAN_21, VERDICT: PLAN_22, or "
    "VERDICT: NO_UNIQUE_PLAN."
)

try:
    register_instrument(GAME_INSTRUMENT, VERDICT_SET,
                        legacy_instruction=GAME_VERDICT_INSTRUCTION)
except ValueError:
    pass  # already registered (module re-import); the selftest checks tokens

# ---------------------------------------------------------------------------
# Leak-control vocabulary
# ---------------------------------------------------------------------------

#: Must never appear (case-insensitive substring) in any text shown to a model.
FORBIDDEN_WORDS = ("cooperate", "defect", "cooperation", "defection",
                   "prisoner", "dilemma", "betray", "trust")

#: Valence-neutral player names with pairwise-distinct initials (option labels
#: are initial + 1/2, so distinct initials keep the labels unambiguous). A and
#: B are avoided because the source files use them.
PLAYER_NAMES = ("Emerson", "Finley", "Greer", "Harper", "Jordan", "Kendall",
                "Lennox", "Morgan", "Noor", "Oakley", "Parker", "Quinn",
                "Rowan", "Sage", "Taylor", "Wren")


def forbidden_hits(text: str) -> list[str]:
    """Forbidden words present in ``text`` (case-insensitive substring)."""
    low = (text or "").lower()
    return [w for w in FORBIDDEN_WORDS if w in low]


# ---------------------------------------------------------------------------
# Solution-concept helpers (ported from the scratchpad game_library.py /
# classify.py; A[i][j] = row payoff, B[i][j] = column payoff at cell (i, j))
# ---------------------------------------------------------------------------

def maximin(A: Matrix, B: Matrix) -> tuple[int, int]:
    vr = max(min(A[i][j] for j in (0, 1)) for i in (0, 1))
    vc = max(min(B[i][j] for i in (0, 1)) for j in (0, 1))
    return vr, vc


def pure_ne(A: Matrix, B: Matrix) -> list[Cell]:
    """Pure Nash cells, weak best responses (>=)."""
    return [(i, j) for (i, j) in CELLS
            if A[i][j] >= A[1 - i][j] and B[i][j] >= B[i][1 - j]]


def strict_ne(A: Matrix, B: Matrix) -> list[Cell]:
    return [(i, j) for (i, j) in CELLS
            if A[i][j] > A[1 - i][j] and B[i][j] > B[i][1 - j]]


def dominant_weak(A: Matrix, B: Matrix) -> tuple[Optional[int], Optional[int]]:
    """Each player's weakly-dominant-with-one-strict action, the canonical
    file's domA/domB definition (classify.py). Equals strict dominance on
    strict-ordinal games."""
    domA = None
    for i in range(2):
        if (all(A[i][j] >= A[1 - i][j] for j in range(2))
                and any(A[i][j] > A[1 - i][j] for j in range(2))):
            domA = i
    domB = None
    for j in range(2):
        if (all(B[i][j] >= B[i][1 - j] for i in range(2))
                and any(B[i][j] > B[i][1 - j] for i in range(2))):
            domB = j
    return domA, domB


def dominant_strict(A: Matrix, B: Matrix) -> tuple[Optional[int], Optional[int]]:
    """game_library.dominant: strictly dominant action per player or None."""
    r = None
    if all(A[0][j] > A[1][j] for j in (0, 1)):
        r = 0
    elif all(A[1][j] > A[0][j] for j in (0, 1)):
        r = 1
    c = None
    if all(B[i][0] > B[i][1] for i in (0, 1)):
        c = 0
    elif all(B[i][1] > B[i][0] for i in (0, 1)):
        c = 1
    return r, c


def _dominates(A: Matrix, B: Matrix, y: Cell, x: Cell) -> bool:
    """y weakly Pareto-dominates x with at least one strict inequality."""
    return (A[y[0]][y[1]] >= A[x[0]][x[1]] and B[y[0]][y[1]] >= B[x[0]][x[1]]
            and (A[y[0]][y[1]] > A[x[0]][x[1]] or B[y[0]][y[1]] > B[x[0]][x[1]]))


def efficient(A: Matrix, B: Matrix) -> list[Cell]:
    """Pareto-efficient cells."""
    return [x for x in CELLS
            if not any(_dominates(A, B, y, x) for y in CELLS if y != x)]


def core(A: Matrix, B: Matrix) -> list[Cell]:
    """Pareto-efficient AND individually rational w.r.t. each maximin."""
    vr, vc = maximin(A, B)
    return [x for x in efficient(A, B)
            if A[x[0]][x[1]] >= vr and B[x[0]][x[1]] >= vc]


def _strict_both(A: Matrix, B: Matrix, c: Cell, n: Cell) -> bool:
    return A[c[0]][c[1]] > A[n[0]][n[1]] and B[c[0]][c[1]] > B[n[0]][n[1]]


def situation_type(c: Mapping[str, Any]) -> str:
    """The canonical file's descriptive ``type`` field (classify.py port)."""
    n = len(c["nash"])
    if c["constant_sum"]:
        return "zero_sum"
    if c["both_best"]:
        if c["both_best"][0] in c["nash"] and n == 1:
            return "harmony"
        if c["both_best"][0] in c["nash"] and n >= 2:
            return "assurance"
        return "harmony_nonnash"
    if n == 1 and c["domA"] is not None and c["domB"] is not None:
        if c["nash"][0] in c["pareto"]:
            return "dominance_efficient"
        return "social_dilemma"
    if n == 1 and (c["domA"] is not None or c["domB"] is not None):
        if c["nash"][0] in c["pareto"]:
            return "one_dominant_efficient"
        return "one_dominant_inefficient"
    if n == 2:
        a, b = c["nash"]
        if (a in c["pareto"]) and (b in c["pareto"]):
            return "two_nash_both_efficient"
        return "two_nash_one_efficient"
    if n == 0:
        return "no_pure_nash"
    return "other"


def gold_rule(A: Matrix, B: Matrix, nash: Sequence[Cell],
              pareto: Sequence[Cell], constant_sum: bool
              ) -> tuple[str, Optional[Cell]]:
    """The derived gold rules, exactly as documented in the module docstring."""
    N = list(nash)
    P = list(pareto)
    if constant_sum:
        return ZERO_SUM, None
    if len(P) == 1:
        p = P[0]
        if len(N) == 1 and N[0] == p:
            return HARMONY, p
        if len(N) >= 2 and p in N:
            return ASSURANCE, p
        if len(N) == 0:
            return CYCLIC_UNIQUE_IDEAL, p
        return DILEMMA_UNIQUE_IDEAL, p
    if N:
        D = [c for c in P if all(_strict_both(A, B, c, n) for n in N)]
        if len(D) == 1:
            return (DILEMMA_BOTH_PREFER if len(N) == 1
                    else ASSURANCE_LIKE_MULTI), D[0]
        if len(D) > 1:
            return AMBIGUOUS_IDEAL_NEEDS_CARDINAL, None
        if len(N) == 1:
            return INCENTIVE_COMPATIBLE_EFFICIENT, N[0]
        return CONFLICT_OF_EQUILIBRIA, None
    return CYCLIC_NO_IDEAL, None


#: Fields of the canonical record that classify() reproduces (everything in
#: the json except pA/pB, which are inputs).
CANONICAL_FIELDS = ("nash", "strict_nash", "domA", "domB", "pareto", "welfare",
                    "both_best", "constant_sum", "nash_pareto", "nash_welfare",
                    "w", "type", "gold_type", "gold_cell")


def classify(pA: Matrix, pB: Matrix) -> dict[str, Any]:
    """Full first-principles classification of one ordinal 2x2 game.

    Returns every canonical-record field (CANONICAL_FIELDS, cells as tuples,
    ``w`` keyed by cell tuple) plus ``maximin`` and ``core`` as extras.
    """
    nash = pure_ne(pA, pB)
    strict = strict_ne(pA, pB)
    domA, domB = dominant_weak(pA, pB)
    pareto = efficient(pA, pB)
    w = {c: pA[c[0]][c[1]] + pB[c[0]][c[1]] for c in CELLS}
    wmax = max(w.values())
    welfare = [c for c in CELLS if w[c] == wmax]
    both_best = [c for c in CELLS if pA[c[0]][c[1]] == 4 and pB[c[0]][c[1]] == 4]
    constant_sum = all(pA[i][j] + pB[i][j] == 5 for (i, j) in CELLS)
    rec: dict[str, Any] = {
        "nash": nash, "strict_nash": strict, "domA": domA, "domB": domB,
        "pareto": pareto, "welfare": welfare, "both_best": both_best,
        "constant_sum": constant_sum,
        "nash_pareto": [c for c in nash if c in pareto],
        "nash_welfare": [c for c in nash if c in welfare],
        "w": w,
    }
    rec["type"] = situation_type(rec)
    gt, gc = gold_rule(pA, pB, nash, pareto, constant_sum)
    rec["gold_type"] = gt
    rec["gold_cell"] = gc
    rec["maximin"] = maximin(pA, pB)
    rec["core"] = core(pA, pB)
    return rec


def _tuplify(x: Any) -> Any:
    if isinstance(x, list):
        return tuple(_tuplify(v) for v in x)
    if isinstance(x, dict):
        return {k: _tuplify(v) for k, v in x.items()}
    return x


def canonical_view(rec: Mapping[str, Any]) -> dict[str, Any]:
    """A classify() record (or a json record) in one comparable shape:
    lists -> tuples, ``w`` keyed by ``str(cell)`` as the json stores it."""
    out: dict[str, Any] = {}
    for k in CANONICAL_FIELDS:
        v = rec[k]
        if k == "w":
            out[k] = {str(tuple(kk)) if not isinstance(kk, str) else kk: vv
                      for kk, vv in v.items()}
        elif k == "gold_cell":
            out[k] = None if v is None else tuple(v)
        else:
            out[k] = _tuplify(v)
    return out


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

#: A data row of the payoff table: label, then two "x \\ y" cells (x, y are a
#: digit or "?"). Label-agnostic so the same parser reads the source files
#: (A1/A2) and rendered items (initial+1/2), and masked tables.
_ROW_RE = re.compile(
    r"^\|\s*([A-Za-z]\d)\s*\|\s*([1-4?])\s*\\\s*([1-4?])\s*\|"
    r"\s*([1-4?])\s*\\\s*([1-4?])\s*\|\s*$", re.M)


def parse_table(text: str) -> tuple[tuple[str, str], tuple[tuple[str, str], ...]]:
    """(row labels, 4 cells in reading order as (row_str, col_str))."""
    rows = _ROW_RE.findall(text or "")
    if len(rows) != 2:
        raise ValueError(f"expected 2 payoff rows, found {len(rows)}")
    labels = (rows[0][0], rows[1][0])
    # reading order: (0,0), (0,1), (1,0), (1,1)
    ordered = ((rows[0][1], rows[0][2]), (rows[0][3], rows[0][4]),
               (rows[1][1], rows[1][2]), (rows[1][3], rows[1][4]))
    return labels, ordered


def parse_payoffs(text: str) -> tuple[Matrix, Matrix]:
    """(pA, pB) from a scenario/table text; raises on a masked table."""
    _, cells = parse_table(text)
    if any(a == "?" or b == "?" for a, b in cells):
        raise ValueError("masked table has no payoff numbers")
    a = [[0, 0], [0, 0]]
    b = [[0, 0], [0, 0]]
    for (i, j), (x, y) in zip(CELLS, cells):
        a[i][j] = int(x)
        b[i][j] = int(y)
    return ((a[0][0], a[0][1]), (a[1][0], a[1][1])), \
           ((b[0][0], b[0][1]), (b[1][0], b[1][1]))


def is_strict_ordinal(M: Matrix) -> bool:
    return sorted(M[0] + M[1]) == [1, 2, 3, 4]


@dataclass(frozen=True)
class Game:
    game_id: str
    pA: Matrix          # row player's payoffs, canonical orientation
    pB: Matrix          # column player's payoffs, canonical orientation
    text: str = field(repr=False)          # the source file, as read
    gold: Mapping[str, Any] = field(repr=False)  # canonical json record

    @property
    def gold_type(self) -> str:
        return str(self.gold["gold_type"])

    @property
    def gold_cell(self) -> Optional[Cell]:
        gc = self.gold["gold_cell"]
        return None if gc is None else (int(gc[0]), int(gc[1]))

    @property
    def region(self) -> str:
        return REGION_OF[self.gold_type]

    @property
    def nash_cells(self) -> tuple[Cell, ...]:
        return tuple((int(c[0]), int(c[1])) for c in self.gold["nash"])


def games_sha256(data_dir: Path = DATA_DIR) -> str:
    h = hashlib.sha256()
    for p in sorted(data_dir.glob("*.txt")):
        h.update(p.name.encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def gold_sha256(gold_path: Path = GOLD_PATH) -> str:
    return hashlib.sha256(gold_path.read_bytes()).hexdigest()


def load_games(data_dir: Path = DATA_DIR, gold_path: Path = GOLD_PATH, *,
               verify_pins: bool = True) -> list[Game]:
    """The 144 games, sorted by id, each with its canonical gold record.

    Refuses (ValueError) if the byte pins fail, if the file set and the gold
    keys differ, or if a file's parsed payoffs differ from its gold record's
    pA/pB -- a Game carrying inconsistent payoffs and gold would be a silent
    corruption of every downstream number.
    """
    if not data_dir.is_dir() or not gold_path.is_file():
        raise FileNotFoundError(
            f"TMGBench data missing: {data_dir} / {gold_path}")
    if verify_pins:
        g = games_sha256(data_dir)
        if g != GAMES_SHA256:
            raise ValueError(f"game files sha256 {g} != pinned {GAMES_SHA256}")
        j = gold_sha256(gold_path)
        if j != GOLD_SHA256:
            raise ValueError(f"gold json sha256 {j} != pinned {GOLD_SHA256}")
    with open(gold_path, encoding="utf-8") as f:
        gold = json.load(f)
    paths = sorted(data_dir.glob("*.txt"))
    ids = [p.stem for p in paths]
    if ids != sorted(gold):
        raise ValueError("game file ids and gold json keys differ")
    if len(ids) != N_GAMES:
        raise ValueError(f"expected {N_GAMES} games, found {len(ids)}")
    out: list[Game] = []
    for p in paths:
        text = p.read_text(encoding="utf-8")
        pA, pB = parse_payoffs(text)
        rec = gold[p.stem]
        if _tuplify(rec["pA"]) != pA or _tuplify(rec["pB"]) != pB:
            raise ValueError(f"game {p.stem}: parsed payoffs differ from gold")
        out.append(Game(game_id=p.stem, pA=pA, pB=pB, text=text, gold=rec))
    return out


def games_by_region(games: Sequence[Game]) -> dict[str, list[Game]]:
    out: dict[str, list[Game]] = {r: [] for r in REGIONS}
    for g in games:
        out[g.region].append(g)
    return out


# ---------------------------------------------------------------------------
# Leak-controlled rendering
# ---------------------------------------------------------------------------

Perm = tuple[int, int]   # display index -> canonical index


def item_permutation(game_id: str, sample_idx: int
                     ) -> tuple[Perm, Perm, str, str]:
    """(row_perm, col_perm, row_name, col_name), deterministic in
    (game_id, sample_idx). Seeded through sha256 so it does not depend on
    PYTHONHASHSEED or on the Random(str) seeding algorithm."""
    digest = hashlib.sha256(
        f"tmg_plan_item|{game_id}|{int(sample_idx)}".encode()).digest()
    rng = random.Random(int.from_bytes(digest, "big"))
    row_perm: Perm = (0, 1) if rng.random() < 0.5 else (1, 0)
    col_perm: Perm = (0, 1) if rng.random() < 0.5 else (1, 0)
    row_name, col_name = rng.sample(PLAYER_NAMES, 2)
    return row_perm, col_perm, row_name, col_name


def plan_for_cell(cell: Cell, row_perm: Perm, col_perm: Perm) -> str:
    """Canonical cell -> PLAN_xy in displayed option numbers."""
    d = row_perm.index(cell[0])
    e = col_perm.index(cell[1])
    return f"PLAN_{d + 1}{e + 1}"


_PLAN_TOKEN_RE = re.compile(r"PLAN_([12])([12])")


def cell_for_plan(verdict: str, row_perm: Perm, col_perm: Perm) -> Optional[Cell]:
    """PLAN_xy -> canonical cell; None for NO_UNIQUE_PLAN / NOVERDICT / other."""
    m = _PLAN_TOKEN_RE.fullmatch(verdict or "")
    if not m:
        return None
    return row_perm[int(m.group(1)) - 1], col_perm[int(m.group(2)) - 1]


def permute(M: Matrix, row_perm: Perm, col_perm: Perm) -> Matrix:
    """The matrix as displayed: display cell (d, e) = M[row_perm[d]][col_perm[e]]."""
    return tuple(tuple(M[row_perm[d]][col_perm[e]] for e in (0, 1))
                 for d in (0, 1))  # type: ignore[return-value]


@dataclass(frozen=True)
class ItemView:
    """One leak-controlled rendering of a game. ``text`` is what a model
    sees (scenario + question + plan legend); everything else is metadata
    for scoring and analysis and must NEVER be placed in a prompt."""

    game_id: str
    sample_idx: int
    masked: bool
    gold_type: str
    region: str
    row_perm: Perm
    col_perm: Perm
    row_name: str
    col_name: str
    row_labels: tuple[str, str]     # displayed option labels, row player
    col_labels: tuple[str, str]     # displayed option labels, column player
    pA: Matrix                      # canonical (true) payoffs, row player
    pB: Matrix                      # canonical (true) payoffs, column player
    display_pA: Matrix              # permuted, as laid out in the table
    display_pB: Matrix
    table_text: str
    scenario_text: str
    question_text: str
    legend_text: str
    gold_verdict: str               # PLAN_xy or NO_UNIQUE_PLAN
    nash_verdicts: frozenset[str]   # PLAN_xy for every pure Nash cell
    row_best_verdict: str           # the plan giving the row player 4
    col_best_verdict: str           # the plan giving the column player 4

    @property
    def item_id(self) -> str:
        return self.game_id

    @property
    def text(self) -> str:
        return f"{self.scenario_text}\n\n{self.question_text}\n\n{self.legend_text}"

    @property
    def prompt_strings(self) -> tuple[str, ...]:
        """Every string of this view that can reach a model."""
        return (self.text, self.scenario_text, self.question_text,
                self.legend_text, self.table_text, self.row_name,
                self.col_name) + self.row_labels + self.col_labels

    def cell_for(self, verdict: str) -> Optional[Cell]:
        return cell_for_plan(verdict, self.row_perm, self.col_perm)

    def payoffs_for(self, verdict: str) -> Optional[tuple[int, int]]:
        """(row payoff, column payoff) at the plan named, or None."""
        c = self.cell_for(verdict)
        if c is None:
            return None
        return self.pA[c[0]][c[1]], self.pB[c[0]][c[1]]

    def is_nash(self, verdict: str) -> Optional[bool]:
        """True/False for a PLAN_xy, None for anything else."""
        if verdict not in PLAN_VERDICTS:
            return None
        return verdict in self.nash_verdicts


def _table(row_name: str, col_name: str, row_labels: tuple[str, str],
           col_labels: tuple[str, str], dA: Matrix, dB: Matrix,
           masked: bool) -> str:
    """Markdown table with NO alignment padding: any padding would be a
    function of the drawn names' lengths, and the masked text must be
    byte-identical across games once names and labels are substituted
    (selftest: surface_template)."""
    def cell(d: int, e: int) -> str:
        return "? \\ ?" if masked else f"{dA[d][e]} \\ {dB[d][e]}"
    lines = [
        f"| {row_name} \\ {col_name} | {col_labels[0]} | {col_labels[1]} |",
        "|---|---|---|",
    ]
    for d in (0, 1):
        lines.append(f"| {row_labels[d]} | {cell(d, 0)} | {cell(d, 1)} |")
    return "\n".join(lines)


def render_item(game: Game, sample_idx: int, *, masked: bool = False) -> ItemView:
    """Leak-controlled rendering, deterministic in (game_id, sample_idx)."""
    row_perm, col_perm, R, C = item_permutation(game.game_id, sample_idx)
    r1, r2 = f"{R[0]}1", f"{R[0]}2"
    c1, c2 = f"{C[0]}1", f"{C[0]}2"
    dA = permute(game.pA, row_perm, col_perm)
    dB = permute(game.pB, row_perm, col_perm)
    table = _table(R, C, (r1, r2), (c1, c2), dA, dB, masked)

    scenario = (
        f"Two players, {R} and {C}, are playing a game. Each of them has two "
        f"options: {R} chooses between {r1} and {r2}, and {C} chooses between "
        f"{c1} and {c2}. They choose at the same time.\n"
        f"The payoff table for their combinations of options is given below. "
        f"Each cell reads \"{R}'s payoff \\ {C}'s payoff\"; for each player a "
        f"larger number means a higher payoff.\n"
        + ("In this version the payoff numbers are withheld: every cell reads "
           "\"? \\ ?\".\n" if masked else "")
        + f"{table}\n\n"
        f"Both {R} and {C} aim to maximize their own payoff."
    )
    question = (
        f"Before they play, {R} and {C} may sign a binding agreement on a joint "
        f"plan -- one option for {R} and one option for {C} -- which both must "
        f"then follow. If there is exactly one joint plan that both players "
        f"should accept, name it. If there is no such plan, or if nothing in "
        f"the table singles out one joint plan over another, answer "
        f"NO_UNIQUE_PLAN."
    )
    legend = (
        f"Joint plans are written PLAN_xy, where x is {R}'s option number and "
        f"y is {C}'s option number:\n"
        f"  PLAN_11  {R} takes {r1} and {C} takes {c1}\n"
        f"  PLAN_12  {R} takes {r1} and {C} takes {c2}\n"
        f"  PLAN_21  {R} takes {r2} and {C} takes {c1}\n"
        f"  PLAN_22  {R} takes {r2} and {C} takes {c2}\n"
        f"  NO_UNIQUE_PLAN  there is no single joint plan that both players "
        f"should accept"
    )

    gc = game.gold_cell
    gold_verdict = NO_UNIQUE_PLAN if gc is None else plan_for_cell(gc, row_perm, col_perm)
    nash_verdicts = frozenset(plan_for_cell(c, row_perm, col_perm)
                              for c in game.nash_cells)
    row_top = next(c for c in CELLS if game.pA[c[0]][c[1]] == 4)
    col_top = next(c for c in CELLS if game.pB[c[0]][c[1]] == 4)
    return ItemView(
        game_id=game.game_id, sample_idx=int(sample_idx), masked=bool(masked),
        gold_type=game.gold_type, region=game.region,
        row_perm=row_perm, col_perm=col_perm, row_name=R, col_name=C,
        row_labels=(r1, r2), col_labels=(c1, c2),
        pA=game.pA, pB=game.pB, display_pA=dA, display_pB=dB,
        table_text=table, scenario_text=scenario, question_text=question,
        legend_text=legend, gold_verdict=gold_verdict,
        nash_verdicts=nash_verdicts,
        row_best_verdict=plan_for_cell(row_top, row_perm, col_perm),
        col_best_verdict=plan_for_cell(col_top, row_perm, col_perm),
    )


def surface_template(view: ItemView) -> str:
    """The view's text with names and option labels replaced by placeholders.
    For masked views this must be identical across all games: the masked
    text then carries no game information at all."""
    # Placeholders carry no digits: a label is initial+digit (e.g. "R1" for
    # Rowan), so a digit-bearing placeholder could itself be re-substituted.
    t = view.text
    for lab, ph in ((view.row_labels[0], "<ROW_ONE>"), (view.row_labels[1], "<ROW_TWO>"),
                    (view.col_labels[0], "<COL_ONE>"), (view.col_labels[1], "<COL_TWO>")):
        t = t.replace(lab, ph)
    t = t.replace(view.row_name, "<ROW>").replace(view.col_name, "<COL>")
    return t


# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------

_NO_UNIQUE_RE = re.compile(r"\bNO[ _\-]UNIQUE[ _\-]PLAN\b", re.I)
#: The first separator is MANDATORY: "PLAN12" is a near-miss like "PLAN_1",
#: not a spacing variant (module docstring, VERDICT PARSING).
_PLAN_SPACING_RE = re.compile(r"\bPLAN[ _\-]([12])[_\-]?([12])\b", re.I)
#: Every legal token as it appears after normalisation, for the hedge check.
_ANY_TOKEN_RE = re.compile(r"PLAN_[12][12]|NO_UNIQUE_PLAN")


def normalize_plan_tokens(text: str) -> str:
    """Collapse spacing/punctuation variants of the exact tokens only:
    'PLAN 12' / 'PLAN-12' / 'PLAN_1_2' -> PLAN_12, 'NO UNIQUE PLAN' ->
    NO_UNIQUE_PLAN. Nothing else is touched; near-misses ('PLAN_1',
    'PLAN 1 2', 'PLAN12') stay near-misses."""
    t = _NO_UNIQUE_RE.sub(NO_UNIQUE_PLAN, text or "")
    return _PLAN_SPACING_RE.sub(lambda m: f"PLAN_{m.group(1)}{m.group(2)}", t)


def parse_plan_verdict(text: str) -> str:
    """One of the five tokens, or NOVERDICT. A stray UNRESOLVED is NOVERDICT:
    it was never offered by GAME_VERDICT_INSTRUCTION. A verdict line naming
    more than one distinct legal token is a hedge and is NOVERDICT (module
    docstring, VERDICT PARSING)."""
    d = extract_verdict_detail(normalize_plan_tokens(text or ""), GAME_INSTRUMENT)
    if d.verdict not in VERDICT_SET:
        return NOVERDICT
    if len(set(_ANY_TOKEN_RE.findall(d.line))) > 1:
        return NOVERDICT
    return d.verdict


def score_plan(verdict: str, gold_verdict: str) -> dict[str, bool]:
    """verdict_format.score_against_gold on this instrument (NO_UNIQUE_PLAN
    is a committal gold and can be right or wrong)."""
    return score_against_gold(verdict, gold_verdict, GAME_INSTRUMENT)


# ---------------------------------------------------------------------------
# Guards. Thresholds mirror run_crowdgold_aita.truncation_report exactly
# (defined locally so this data module stays import-light).
# ---------------------------------------------------------------------------

MAX_TRUNCATION_SHARE = 0.05
MAX_NOVERDICT_SHARE = 0.05


def parse_guard(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-(model, scaffold, arm) truncation and NOVERDICT shares with a pass
    flag. Rows need ``verdict``; ``truncated`` and ``output_len`` are read
    when present."""
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(str(r.get("model", "")), str(r.get("scaffold", "")),
                str(r.get("arm", "")))].append(r)
    cells: dict[str, dict[str, Any]] = {}
    worst_t = worst_n = 0.0
    for (m, sc, arm), rs in sorted(groups.items()):
        t = sum(int(r.get("truncated", 0) or 0) for r in rs) / len(rs)
        nv = sum(1 for r in rs if r.get("verdict") == NOVERDICT) / len(rs)
        cells[f"{m}|{sc}|{arm}"] = {
            "n": len(rs), "truncated": round(t, 4), "noverdict": round(nv, 4),
            "mean_chars": round(sum(int(r.get("output_len", 0) or 0)
                                    for r in rs) / len(rs), 1),
        }
        worst_t, worst_n = max(worst_t, t), max(worst_n, nv)
    return {
        "cells": cells,
        "worst_truncation": round(worst_t, 4),
        "worst_noverdict": round(worst_n, 4),
        "max_truncation_share": MAX_TRUNCATION_SHARE,
        "max_noverdict_share": MAX_NOVERDICT_SHARE,
        "pass": worst_t <= MAX_TRUNCATION_SHARE and worst_n <= MAX_NOVERDICT_SHARE,
    }


def print_parse_guard(rep: Mapping[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("TRUNCATION / PARSE GUARD (PLAN verdicts)")
    print("=" * 72)
    print(f"{'model|scaffold|arm':<58}{'n':>4}{'trunc':>8}{'NOVER':>8}{'chars':>8}")
    for k, v in rep["cells"].items():
        flag = ("  <-- FAILS" if (v["truncated"] > MAX_TRUNCATION_SHARE
                                  or v["noverdict"] > MAX_NOVERDICT_SHARE) else "")
        print(f"{k:<58}{v['n']:>4}{v['truncated']:>8.1%}{v['noverdict']:>8.1%}"
              f"{v['mean_chars']:>8.0f}{flag}")
    if not rep["pass"]:
        print(f"\n  *** GUARD FAILED. Worst truncation {rep['worst_truncation']:.1%} "
              f"(limit {MAX_TRUNCATION_SHARE:.0%}), worst NOVERDICT "
              f"{rep['worst_noverdict']:.1%} (limit {MAX_NOVERDICT_SHARE:.0%}).")
        print("  Raise the failing round's token cap and re-run before reading "
              "any accuracy, Nash-pull or grip number off this run.")
    else:
        print(f"\n  guard PASSED (worst truncation {rep['worst_truncation']:.1%}, "
              f"worst NOVERDICT {rep['worst_noverdict']:.1%})")


# ---------------------------------------------------------------------------
# Masked (hypothesis-only) floor check: item-clustered bootstrap, the same
# idiom as analyze_dilemma_grip._boot_ci.
# ---------------------------------------------------------------------------

def _boot_ci(by_item: Mapping[str, list], stat: Callable[[list], Optional[float]],
             *, draws: int, seed: int, alpha: float = 0.05):
    ids = sorted(by_item)
    rng = random.Random(seed)
    point = stat([x for i in ids for x in by_item[i]])
    vals = sorted(v for v in (
        stat([x for i in (ids[rng.randrange(len(ids))] for _ in ids)
              for x in by_item[i]]) for _ in range(draws)) if v is not None)
    if not vals:
        return point, None, None
    return (point, vals[int(alpha / 2 * len(vals))],
            vals[min(int((1 - alpha / 2) * len(vals)), len(vals) - 1)])


def masked_floor_report(records: Sequence[Mapping[str, Any]], *,
                        draws: int = 2000, seed: int = 7) -> dict[str, Any]:
    """Is masked accuracy at the base-rate floor? Records carry ``item_id``,
    ``gold_verdict``, ``verdict``. The floor is the majority-class rate of
    the gold over these records (the best constant guess); ``chance`` is the
    uniform 5-way rate. ``leak_suspected`` when the item-clustered 95% CI
    lower bound of accuracy exceeds the floor."""
    recs = [r for r in records if r.get("gold_verdict") in VERDICT_SET]
    by_item: dict[str, list] = defaultdict(list)
    for r in recs:
        by_item[str(r["item_id"])].append(int(r.get("verdict") == r["gold_verdict"]))
    if not recs:
        return {"n": 0, "n_items": 0, "accuracy": None, "lo": None, "hi": None,
                "floor": None, "chance": 1 / len(VERDICT_SET),
                "leak_suspected": False}

    def stat(xs: list) -> Optional[float]:
        return sum(xs) / len(xs) if xs else None

    acc, lo, hi = _boot_ci(by_item, stat, draws=draws, seed=seed)
    gold_counts = Counter(r["gold_verdict"] for r in recs)
    floor = max(gold_counts.values()) / len(recs)
    return {
        "n": len(recs), "n_items": len(by_item), "accuracy": acc,
        "lo": lo, "hi": hi, "floor": floor, "chance": 1 / len(VERDICT_SET),
        "gold_counts": dict(gold_counts),
        "verdict_counts": dict(Counter(r.get("verdict") for r in recs)),
        "leak_suspected": lo is not None and lo > floor,
    }


def print_masked_floor_report(rep: Mapping[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("MASKED (HYPOTHESIS-ONLY) FLOOR CHECK")
    print("=" * 72)
    if not rep["n"]:
        print("  no scoreable records")
        return
    lo = "n/a" if rep["lo"] is None else f"{rep['lo']:.3f}"
    hi = "n/a" if rep["hi"] is None else f"{rep['hi']:.3f}"
    print(f"  n={rep['n']} items={rep['n_items']} accuracy={rep['accuracy']:.3f} "
          f"[{lo}, {hi}]  floor(majority gold)={rep['floor']:.3f}  "
          f"chance(5-way)={rep['chance']:.3f}")
    print(f"  gold: {rep['gold_counts']}")
    print(f"  verdicts: {rep['verdict_counts']}")
    if rep["leak_suspected"]:
        print("\n  *** LEAK SUSPECTED: masked accuracy CI sits above the "
              "base-rate floor. The item text carries game information; do "
              "not read any unmasked result until this is fixed.")
    else:
        print("\n  masked accuracy is at the floor: surface form carries nothing")


# ---------------------------------------------------------------------------
# Cache-namespace helper for the runners built on this module
# ---------------------------------------------------------------------------

def cache_files_with_prefix(prefix: str, out_dir: Path = OUT_DIR) -> list[Path]:
    """Existing cache files whose name starts with ``prefix``. Caches are
    keyed by NAME, not content: a runner introducing a new tag / scaffold /
    role id must assert this is empty in its selftest."""
    if not out_dir.is_dir():
        return []
    return sorted(p for p in out_dir.iterdir() if p.name.startswith(prefix))


# ---------------------------------------------------------------------------
# Selftest (offline; no API, no network, no spend)
# ---------------------------------------------------------------------------

_DIGIT_CELL_RE = re.compile(r"[1-4]\s*\\\s*[1-4]")
_MASKED_CELL_RE = re.compile(r"\?\s*\\\s*\?")
N_SELFTEST_SAMPLES = 8


def _selftest() -> int:
    fails: list[str] = []

    def check(name: str, cond: bool) -> None:
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    # ---- data pins -------------------------------------------------------
    check(f"data dir present with {N_GAMES} game files",
          DATA_DIR.is_dir() and len(list(DATA_DIR.glob("*.txt"))) == N_GAMES)
    check("game files sha256 matches pin", games_sha256() == GAMES_SHA256)
    check("gold json sha256 matches pin", gold_sha256() == GOLD_SHA256)

    try:
        games = load_games()
    except Exception as e:  # noqa: BLE001
        check(f"load_games() ({type(e).__name__}: {e})", False)
        print(f"\n{len(fails)} FAILED")
        return 1
    check(f"load_games returns {N_GAMES} games", len(games) == N_GAMES)
    check("game ids are unique and sorted",
          [g.game_id for g in games] == sorted({g.game_id for g in games}))

    with open(GOLD_PATH, encoding="utf-8") as f:
        gold_json = json.load(f)

    # ---- parsed payoffs vs json ------------------------------------------
    bad_pay = [g.game_id for g in games
               if (parse_payoffs(g.text) != (_tuplify(gold_json[g.game_id]["pA"]),
                                             _tuplify(gold_json[g.game_id]["pB"])))]
    check(f"txt-parsed pA/pB equal the json's pA/pB on {N_GAMES - len(bad_pay)}/"
          f"{N_GAMES}" + (f" (mismatch: {bad_pay})" if bad_pay else ""),
          not bad_pay)
    check("every payoff matrix is a strict ordinal permutation of 1..4",
          all(is_strict_ordinal(g.pA) and is_strict_ordinal(g.pB) for g in games))

    # ---- classify reproduces the canonical gold ---------------------------
    mism_gold: list[str] = []
    mism_fields: Counter = Counter()
    mism_ids_by_field: dict[str, list[str]] = defaultdict(list)
    for g in games:
        got = canonical_view(classify(g.pA, g.pB))
        want = canonical_view(gold_json[g.game_id])
        if got["gold_type"] != want["gold_type"] or got["gold_cell"] != want["gold_cell"]:
            mism_gold.append(f"{g.game_id}: got {got['gold_type']}/{got['gold_cell']} "
                             f"want {want['gold_type']}/{want['gold_cell']}")
        for k in CANONICAL_FIELDS:
            if got[k] != want[k]:
                mism_fields[k] += 1
                mism_ids_by_field[k].append(g.game_id)
    check(f"classify reproduces gold_type AND gold_cell on "
          f"{N_GAMES - len(mism_gold)}/{N_GAMES}", not mism_gold)
    for line in mism_gold:
        print(f"      MISMATCH {line}")
    check("classify reproduces every other canonical field (nash, strict_nash, "
          "domA, domB, pareto, welfare, both_best, constant_sum, nash_pareto, "
          "nash_welfare, w, type) on 144/144"
          + (f" -- mismatches {dict(mism_fields)}" if mism_fields else ""),
          not mism_fields)
    for k, ids in mism_ids_by_field.items():
        print(f"      field {k}: {ids[:12]}")
    check("weak-with-one-strict dominance equals strict dominance on every game",
          all(dominant_weak(g.pA, g.pB) == dominant_strict(g.pA, g.pB) for g in games))
    check("weak and strict pure Nash coincide on every game",
          all(pure_ne(g.pA, g.pB) == strict_ne(g.pA, g.pB) for g in games))

    counts = Counter(g.gold_type for g in games)
    check(f"exactly the seven registered gold types with registered counts "
          f"{dict(counts)}", dict(counts) == REGISTERED_TYPE_COUNTS)
    rcounts = Counter(g.region for g in games)
    check(f"regions value/neutral/cannot_assist = "
          f"{rcounts[REGION_VALUE]}/{rcounts[REGION_NEUTRAL]}/"
          f"{rcounts[REGION_CANNOT_ASSIST]}", dict(rcounts) == REGISTERED_REGION_COUNTS)
    check("gold_cell is null exactly on the cannot_assist region",
          all((g.gold_cell is None) == (g.region == REGION_CANNOT_ASSIST)
              for g in games))
    check("canonical gold cell is the fixed position (0,1) on every DILEMMA / "
          "ASSURANCE / HARMONY game (the positional leak the permutation fixes)",
          all(g.gold_cell == (0, 1) for g in games
              if g.gold_type in (DILEMMA_BOTH_PREFER, ASSURANCE, HARMONY)))
    check("gold cell is a pure Nash cell on every neutral-region game and NOT "
          "a Nash cell on any DILEMMA_BOTH_PREFER game",
          all(g.gold_cell in g.nash_cells for g in games if g.region == REGION_NEUTRAL)
          and all(g.gold_cell not in g.nash_cells for g in games
                  if g.gold_type == DILEMMA_BOTH_PREFER))

    # ---- instrument ------------------------------------------------------
    check("instrument registered with verdict_format: five tokens + UNRESOLVED",
          INSTRUMENTS.get(GAME_INSTRUMENT) == VERDICT_SET + (UNRESOLVED,))
    check("GAME_VERDICT_INSTRUCTION equals verdict_format's derived wording "
          "(allow_unresolved=False)",
          verdict_instruction(GAME_INSTRUMENT, allow_unresolved=False)
          == GAME_VERDICT_INSTRUCTION)
    check("GAME_VERDICT_INSTRUCTION names all five tokens and never UNRESOLVED",
          all(f"VERDICT: {t}" in GAME_VERDICT_INSTRUCTION for t in VERDICT_SET)
          and UNRESOLVED not in GAME_VERDICT_INSTRUCTION)
    check("verdict instruction is free of forbidden words",
          not forbidden_hits(GAME_VERDICT_INSTRUCTION))
    check("player-name list has 16 names with pairwise-distinct initials, none "
          "containing a forbidden word",
          len(PLAYER_NAMES) == 16
          and len({n[0] for n in PLAYER_NAMES}) == len(PLAYER_NAMES)
          and not any(forbidden_hits(n) for n in PLAYER_NAMES))

    # ---- rendering: forbidden words, determinism, round trips --------------
    views: dict[tuple[str, int], ItemView] = {}
    masked_views: dict[tuple[str, int], ItemView] = {}
    for g in games:
        for idx in range(N_SELFTEST_SAMPLES):
            views[(g.game_id, idx)] = render_item(g, idx)
            masked_views[(g.game_id, idx)] = render_item(g, idx, masked=True)
    all_views = list(views.values()) + list(masked_views.values())
    leaks = [(v.game_id, v.sample_idx, v.masked, forbidden_hits(s))
             for v in all_views for s in v.prompt_strings if forbidden_hits(s)]
    check(f"no forbidden word in any rendered prompt string "
          f"({len(all_views)} views x {len(all_views[0].prompt_strings)} strings)"
          + (f" -- first leak {leaks[0]}" if leaks else ""), not leaks)
    check("rendering is deterministic in (game_id, sample_idx)",
          all(render_item(g, 3) == views[(g.game_id, 3)] for g in games[:20]))
    check("sample_idx changes the rendering (permutation or names differ) for "
          "every game between idx 0 and some idx in 1..7",
          all(any(views[(g.game_id, i)].text != views[(g.game_id, 0)].text
                  for i in range(1, N_SELFTEST_SAMPLES)) for g in games))

    rt_bad = []
    for v in views.values():
        g = next(x for x in games if x.game_id == v.game_id)
        try:
            dA, dB = parse_payoffs(v.table_text)
        except ValueError:
            rt_bad.append((v.game_id, v.sample_idx, "unparseable"))
            continue
        ok = (dA == v.display_pA and dB == v.display_pB
              and dA == permute(g.pA, v.row_perm, v.col_perm)
              and all(v.pA[c[0]][c[1]] == dA[v.row_perm.index(c[0])][v.col_perm.index(c[1])]
                      for c in CELLS)
              and all(cell_for_plan(plan_for_cell(c, v.row_perm, v.col_perm),
                                    v.row_perm, v.col_perm) == c for c in CELLS)
              and v.cell_for(v.gold_verdict) == g.gold_cell
              and {v.cell_for(p) for p in v.nash_verdicts} == set(g.nash_cells)
              and v.payoffs_for(v.row_best_verdict)[0] == 4
              and v.payoffs_for(v.col_best_verdict)[1] == 4
              and v.cell_for(NO_UNIQUE_PLAN) is None
              and parse_table(v.table_text)[0] == v.row_labels)
        if not ok:
            rt_bad.append((v.game_id, v.sample_idx, "mismatch"))
    check("round trip: re-parsed rendered table == permuted payoffs; "
          "cell->plan->cell identity; gold/nash/best verdicts map back to the "
          "canonical cells" + (f" -- first bad {rt_bad[0]}" if rt_bad else ""),
          not rt_bad)
    check("gold_verdict is NO_UNIQUE_PLAN exactly on cannot_assist views",
          all((v.gold_verdict == NO_UNIQUE_PLAN) == (v.region == REGION_CANNOT_ASSIST)
              for v in views.values()))
    check("every rendered verdict-bearing field is a legal token",
          all(v.gold_verdict in VERDICT_SET and v.nash_verdicts <= set(PLAN_VERDICTS)
              and v.row_best_verdict in PLAN_VERDICTS
              and v.col_best_verdict in PLAN_VERDICTS for v in views.values()))

    value_games = [g for g in games if g.region == REGION_VALUE]
    const_value = [g.game_id for g in value_games
                   if len({views[(g.game_id, i)].gold_verdict
                           for i in range(N_SELFTEST_SAMPLES)}) == 1]
    check(f"gold verdict position is not constant across sample_idx 0..7 for any "
          f"of the {len(value_games)} value-region games"
          + (f" -- constant: {const_value}" if const_value else ""),
          len(value_games) == 16 and not const_value)
    committal = [g for g in games if g.gold_cell is not None]
    const_all = [g.game_id for g in committal
                 if len({views[(g.game_id, i)].gold_verdict
                         for i in range(N_SELFTEST_SAMPLES)}) == 1]
    check(f"...and for none of the {len(committal)} committal games"
          + (f" -- constant: {const_all}" if const_all else ""), not const_all)
    pos_value = Counter(views[(g.game_id, i)].gold_verdict
                        for g in value_games for i in range(N_SELFTEST_SAMPLES))
    check(f"pooled over value-region views all four positions occur "
          f"{dict(pos_value)}", set(pos_value) == set(PLAN_VERDICTS))
    pos_all = Counter(views[(g.game_id, i)].gold_verdict
                      for g in committal for i in range(N_SELFTEST_SAMPLES))
    tot = sum(pos_all.values())
    check(f"pooled over committal views each position has >= 15% share "
          f"{ {k: round(v / tot, 3) for k, v in sorted(pos_all.items())} }",
          set(pos_all) == set(PLAN_VERDICTS)
          and all(v / tot >= 0.15 for v in pos_all.values()))

    # ---- masked mode -----------------------------------------------------
    check("masked text has no payoff digits in payoff positions and exactly "
          "four '? \\ ?' cells",
          all(not _DIGIT_CELL_RE.search(v.text)
              and len(_MASKED_CELL_RE.findall(v.table_text)) == 4
              and all(c == ("?", "?") for c in parse_table(v.table_text)[1])
              for v in masked_views.values()))
    check("unmasked text has exactly four digit cells and no '?' cells",
          all(len(_DIGIT_CELL_RE.findall(v.table_text)) == 4
              and not _MASKED_CELL_RE.search(v.text) for v in views.values()))
    check("masked view shares permutation, names, labels and gold with its "
          "unmasked twin",
          all((m.row_perm, m.col_perm, m.row_name, m.col_name, m.row_labels,
               m.col_labels, m.gold_verdict, m.nash_verdicts)
              == (u.row_perm, u.col_perm, u.row_name, u.col_name, u.row_labels,
                  u.col_labels, u.gold_verdict, u.nash_verdicts)
              for (k, m), u in zip(masked_views.items(),
                                   (views[k] for k in masked_views))))
    templates = {surface_template(v) for v in masked_views.values()}
    check(f"masked surface template is identical across all {len(masked_views)} "
          f"masked views (carries no game information): {len(templates)} template(s)",
          len(templates) == 1)
    check("parse_payoffs refuses a masked table",
          all(_raises(parse_payoffs, v.table_text) for v in list(masked_views.values())[:5]))

    # ---- verdict parsing ---------------------------------------------------
    cases = (
        ("The plan is clear.\n\nVERDICT: PLAN_12", PLAN_12),
        ("**Verdict: plan 21**", PLAN_21),
        ("VERDICT: PLAN-22.", PLAN_22),
        ("verdict: plan_1_1", PLAN_11),
        ("VERDICT: NO_UNIQUE_PLAN", NO_UNIQUE_PLAN),
        ("VERDICT: NO UNIQUE PLAN", NO_UNIQUE_PLAN),
        ("> VERDICT: no-unique-plan", NO_UNIQUE_PLAN),
        ("I lean PLAN_11 but\n\nVERDICT: PLAN_22", PLAN_22),
        ("VERDICT: PLAN_12 (M1 with Q2)", PLAN_12),
        ("VERDICT: PLAN_1", NOVERDICT),
        ("VERDICT: PLAN_13", NOVERDICT),
        ("VERDICT: PLAN 1 2", NOVERDICT),
        ("VERDICT: PLAN12", NOVERDICT),
        ("VERDICT: UNRESOLVED", NOVERDICT),
        ("VERDICT: PLAN_11 or VERDICT: PLAN_12", NOVERDICT),
        ("VERDICT: PLAN_12 or PLAN_21", NOVERDICT),
        ("VERDICT: PLAN_12 and PLAN_21 are both acceptable", NOVERDICT),
        ("VERDICT: PLAN_12; NO_UNIQUE_PLAN", NOVERDICT),
        ("VERDICT: PLAN_21, since PLAN_12 leaves R at 1", NOVERDICT),
        ("VERDICT: NO UNIQUE PLAN, PLAN 12 and PLAN 21 tie", NOVERDICT),
        (GAME_VERDICT_INSTRUCTION, NOVERDICT),
        ("They should both take option 1.", NOVERDICT),
        ("", NOVERDICT),
    )
    bad_cases = [(t, parse_plan_verdict(t), want) for t, want in cases
                 if parse_plan_verdict(t) != want]
    check(f"parse_plan_verdict: {len(cases) - len(bad_cases)}/{len(cases)} cases"
          + (f" -- first bad {bad_cases[0]!r}" if bad_cases else ""), not bad_cases)
    check("parse_plan_verdict only ever returns a legal token or NOVERDICT",
          all(parse_plan_verdict(t) in VERDICT_SET + (NOVERDICT,) for t, _ in cases))
    check("score_plan: NO_UNIQUE_PLAN gold scores correct/incorrect, "
          "NOVERDICT is noncommittal and never correct",
          score_plan(NO_UNIQUE_PLAN, NO_UNIQUE_PLAN) == {"correct": True, "noncommittal": False}
          and score_plan(PLAN_11, NO_UNIQUE_PLAN) == {"correct": False, "noncommittal": False}
          and score_plan(NOVERDICT, PLAN_11) == {"correct": False, "noncommittal": True}
          and score_plan(PLAN_11, PLAN_11)["correct"])

    # ---- guard -----------------------------------------------------------
    clean = [{"model": "m", "scaffold": "s", "arm": "a", "verdict": PLAN_11,
              "truncated": 0, "output_len": 100} for _ in range(20)]
    dirty = clean[:18] + [{"model": "m", "scaffold": "s", "arm": "a",
                           "verdict": NOVERDICT, "truncated": 1, "output_len": 5}
                          for _ in range(2)]
    check("parse_guard passes a clean set", parse_guard(clean)["pass"])
    rep = parse_guard(dirty)
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_parse_guard(rep)
    check("parse_guard fails at 10% NOVERDICT/truncation and prints GUARD FAILED",
          not rep["pass"] and "GUARD FAILED" in buf.getvalue())

    # ---- masked floor report ---------------------------------------------
    rng = random.Random(11)
    at_floor = [{"item_id": v.game_id, "gold_verdict": v.gold_verdict,
                 "verdict": rng.choice(VERDICT_SET)} for v in views.values()]
    r_floor = masked_floor_report(at_floor, draws=300, seed=1)
    want_floor = max(Counter(r["gold_verdict"] for r in at_floor).values()) / len(at_floor)
    check(f"masked_floor_report: random guessing is not flagged "
          f"(acc {r_floor['accuracy']:.3f}, floor {r_floor['floor']:.3f} = "
          f"majority gold share)",
          not r_floor["leak_suspected"] and abs(r_floor["floor"] - want_floor) < 1e-12)
    perfect = [{"item_id": v.game_id, "gold_verdict": v.gold_verdict,
                "verdict": v.gold_verdict} for v in views.values()]
    r_perf = masked_floor_report(perfect, draws=300, seed=1)
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_masked_floor_report(r_perf)
    check("masked_floor_report: perfect masked accuracy is flagged LEAK SUSPECTED",
          r_perf["leak_suspected"] and "LEAK SUSPECTED" in buf.getvalue())

    # ---- cache namespace helper --------------------------------------------
    check("cache_files_with_prefix finds nothing for an unused prefix",
          cache_files_with_prefix("tmg_plan_selftest_never_used_") == [])
    check("this module introduces no cache tag or role id (no 'tmg_' cache "
          "files exist; runners must assert their own tags)",
          cache_files_with_prefix("tmg_") == [])

    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


def _raises(fn: Callable, *a: Any) -> bool:
    try:
        fn(*a)
    except ValueError:
        return True
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _summary(games: Sequence[Game]) -> None:
    print(f"{N_GAMES} TMGBench classic games  data={DATA_DIR}")
    print(f"{'gold_type':<32}{'region':<16}{'n':>4}  ids")
    for gt in GOLD_TYPES:
        ids = [g.game_id for g in games if g.gold_type == gt]
        print(f"{gt:<32}{REGION_OF[gt]:<16}{len(ids):>4}  "
              f"{' '.join(ids[:10])}{' ...' if len(ids) > 10 else ''}")
    print(f"\ninstrument {GAME_INSTRUMENT}: {VERDICT_SET}")
    print(GAME_VERDICT_INSTRUCTION)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="TMGBench classic 2x2 games: gold, leak-controlled "
                    "rendering, PLAN verdict instrument (no API calls)")
    ap.add_argument("--selftest", action="store_true",
                    help="offline self-check; no API keys, no network")
    ap.add_argument("--summary", action="store_true",
                    help="per-gold_type / region counts")
    ap.add_argument("--show", default=None, metavar="GAME_ID",
                    help="print one rendered item (with metadata)")
    ap.add_argument("--sample-idx", type=int, default=0)
    ap.add_argument("--masked", action="store_true")
    ap.add_argument("--extract", default=None, metavar="TEXT",
                    help="parse a PLAN verdict from TEXT")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()
    did = False
    if args.extract is not None:
        did = True
        print(parse_plan_verdict(args.extract))
    if args.summary or args.show:
        games = load_games()
        if args.summary:
            did = True
            _summary(games)
        if args.show:
            did = True
            g = next((x for x in games if x.game_id == args.show), None)
            if g is None:
                print(f"unknown game id {args.show!r}")
                return 2
            v = render_item(g, args.sample_idx, masked=args.masked)
            print(v.text)
            print("\n--- metadata (never shown to a model) ---")
            print(f"game {v.game_id} idx {v.sample_idx} masked={v.masked} "
                  f"gold_type={v.gold_type} region={v.region}")
            print(f"row_perm={v.row_perm} col_perm={v.col_perm} "
                  f"gold={v.gold_verdict} nash={sorted(v.nash_verdicts)} "
                  f"row_best={v.row_best_verdict} col_best={v.col_best_verdict}")
    if not did:
        ap.print_help(sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
