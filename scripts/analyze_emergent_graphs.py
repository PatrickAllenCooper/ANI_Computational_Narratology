"""
scripts/analyze_emergent_graphs.py. The graphs that EMERGE from the cached
deliberation data, drawn from the votes and rows CSVs at zero spend.

THE QUESTION

The deployed protocol is a designed graph (three seats, a moderator, a judge).
The theory in papers/embodied_sensor/headline_theory.html sections 2.1 to 2.5
says the value of that graph is in one edge, the pair of opposed advocates,
whose objections are anti-correlated because any committal one-loser verdict
undermines exactly one of the two stakes. This module measures the graph the
data actually contain rather than the one the protocol draws. For every cached
three-seat community it builds an OBJECTION GRAPH over the three seats. A node
is a seat weighted by its r3 objection rate (the votes CSV bit `objected_r3`,
which is a REJECT or ACCEPT_WITH_MODIFICATION label on the moderator's first
synthesis). An edge is a pair of seats weighted by the phi coefficient between
their objection indicators. Every rate and every phi carries an item-clustered
percentile bootstrap interval. The graph is computed on the full codable
population and again inside the one-loser and both-party verdict strata, so
the reader can see the edge that the theory says exists only under one-loser
verdicts appear and disappear.

A second readout is the WHO-LOSES graph. For each verdict type the module
reports which seat objects, reproducing the Addendum 16.12 finding that the
writer's advocate objects on 0.992 of YTA verdicts and the counterparty on
0.988 of NTA verdicts. A third readout is the SENSOR block, the stake-blind
counter (two or more seats object) with its fire rate and its error lift inside
each stratum, which reproduces the 16.15.1 one-loser ratio of 9.83x.

COMMUNITIES

The four cells of the 16.10 roles x edges factorial on grok (cg_deliberation,
cg_deliberation_noedge, cg_deliberation_identical,
cg_deliberation_identical_noedge), the nano and haiku AITA communities
(cg_deliberation_nano, cg_deliberation_haiku_fixed), the Scruples Dilemmas run
(cg_dilemma_stage1), the games binding Stage B1 communities (cg_game_neutral,
cg_game_planner) and the non-binding Stage C1 communities (cgnb_game_neutral,
cgnb_game_planner). Four further games communities are carried as extras and
flagged `requested: false`, the 16.13d coordination games (cgnb_game_neutral_d,
cgnb_game_planner_d) and the B2 extent map (cg_game_neutral_b2,
cg_game_planner_b2), because they are the games with a one-loser structure and
the games outside the value region respectively.

POPULATIONS AND CONVENTIONS

Three populations are reported per community because the record uses all
three. The RAW population is every debate and every vote, the 16.10 convention
under which the fire rate is 0.196 and the seat marginals are 0.430, 0.690 and
0.073. The S2-CODABLE population is every debate whose final integrated
verdict is codable, the 16.12 convention (1,678 grok debates), on which the
objection graphs, the strata and the who-loses table are computed so that the
one-loser and both-party strata partition the population. The S1-AND-S2-CODABLE
population is the 16.15 convention (1,677 grok debates), on which the sensor
block is computed so that its one-loser ratio is the registered 9.83x.
AITA verdicts are scored by the registered five-token collapse
(analyze_crowdgold_sdt.code_response), never by string equality. Dilemmas
verdicts are scored by token equality (the analyze_dilemma_grip convention).
Games verdicts are scored by the rows CSV `correct` column, which the games
analyzers wrote against the registered gold of each regime.

ESTIMATOR. Item-clustered percentile bootstrap, 4,000 draws, seed 13, the
resampling sequence of analyze_unembodied_ablation._boot reproduced draw for
draw (sorted item ids, random.Random(seed), one randrange per item per draw)
and evaluated on per-item count vectors so that a community with fifty
statistics runs in seconds. The selftest asserts that the two implementations
return identical points and identical interval bounds. The resampling frame is
the FULL population and every stratum restriction is applied inside the draw,
the convention fixed after the 16.15 audit. A draw on which a statistic is
undefined (an empty stratum, a phi with a zero margin) is dropped from that
statistic's interval, which is _boot's own convention.

GUARDS, per community, checked before any number is written. Every debate
carries all three seats' votes on at least 0.99 of the population. The rows
CSV `n_objectors` equals the per-seat `objected_r3` sum on at least 0.99. The
mean `n_r3_unparsed` per debate is at most 0.05. The S2-codable share is
recorded and compared with 0.95. A community whose structural guards fail is
written with status UNREAD and no graph. A community whose only failing guard
is the codable share (the Dilemmas run at 0.907, whose UNRESOLVED token has
no registered exemption) is written with status READ_WITH_CAUTION and the
share printed beside every block, because the objection graph does not depend
on the verdict being codable while the strata and the sensor lift do.

AGREEMENT. The artefact records cell-by-cell agreement with the existing JSON
artefacts that the record cites (topology_2x2_analysis.json for the 16.10 fire
rates and marginals, flooding_analysis.json for the 16.12 who-loses rates and
phis, router_decomposition.json for the 16.15.1 ratio and the routing gain)
and with the quoted record values themselves.

Usage
-----
  python -m scripts.analyze_emergent_graphs --selftest
  python -m scripts.analyze_emergent_graphs
  python -m scripts.analyze_emergent_graphs --draws 400 --seed 13 --only grok_aita
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from scripts.analyze_actuator_ladder import correct as aita_correct
from scripts.analyze_crowdgold_sdt import code_response
from scripts.run_phase1_quartet import OUT_DIR

ANALYSIS_PATH = OUT_DIR / "emergent_graphs.json"
TOPOLOGY_PATH = OUT_DIR / "topology_2x2_analysis.json"
FLOODING_PATH = OUT_DIR / "flooding_analysis.json"
ROUTER_PATH = OUT_DIR / "router_decomposition.json"

DEFAULT_DRAWS = 4000
DEFAULT_SEED = 13
GUARD_FAILED = "GUARD FAILED"
MIN_SEAT_COVERAGE = 0.99
MIN_OBJECTOR_CONSISTENCY = 0.99
MAX_MEAN_R3_UNPARSED = 0.05
MIN_CODABLE_SHARE = 0.95
RATE_TOL = 0.0005 + 1e-9        # record values are quoted to three decimals
EXACT_TOL = 1e-9                # same computation on the same rows

SEAT_POSITIONS = ("advocate_1", "advocate_2", "third")
PAIRS = (("advocate_1", "advocate_2"), ("advocate_1", "third"), ("advocate_2", "third"))


# ---------------------------------------------------------------------------
# Instruments
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Instrument:
    name: str
    codable: Callable[[str], bool]
    correct: Callable[[dict], Optional[bool]]          # from the row dict (verdict, gold, ...)
    strata: dict                                        # name -> (description, predicate on debate)
    who_loses: dict                                     # group token -> description of who loses
    who_loses_key: Callable[[dict], str]                # debate -> group token
    verdict_note: str


def _aita_correct_row(r: dict) -> Optional[bool]:
    return aita_correct(r["verdict"], r["gold_verdict"])


AITA = Instrument(
    name="aita",
    codable=lambda v: code_response(v, "published") is not None,
    correct=_aita_correct_row,
    strata={
        "one_loser": ("final verdict YTA or NTA, exactly one advocate's stake is undermined",
                      lambda d: d["s2"] in ("YTA", "NTA")),
        "both_party": ("final verdict ESH or NAH, a both-party verdict",
                       lambda d: d["s2"] in ("ESH", "NAH")),
    },
    who_loses={
        "YTA": "the writer is at fault, so the writer's advocate (advocate_1) loses",
        "NTA": "the writer is not at fault, so the counterparty (advocate_2) loses",
        "ESH": "everyone at fault, a both-party verdict (collapses to at fault)",
        "NAH": "no one at fault, a both-party verdict (collapses to not at fault)",
    },
    who_loses_key=lambda d: d["s2"],
    verdict_note="five-token AITA verdict scored by the registered code_response collapse",
)

DILEMMA_TYPES = ("ACTION_A", "ACTION_B")
DILEMMAS = Instrument(
    name="dilemmas",
    codable=lambda v: v in DILEMMA_TYPES,
    correct=lambda r: (r["verdict"] == r["gold_verdict"]) if r["verdict"] in DILEMMA_TYPES else None,
    strata={
        "one_loser": ("every codable verdict names one person's action as the more wrong, "
                      "so exactly one advocate loses by construction",
                      lambda d: d["s2"] in DILEMMA_TYPES),
    },
    who_loses={
        "ACTION_A": "person A's action was the more wrong, so person A's advocate (advocate_1) loses",
        "ACTION_B": "person B's action was the more wrong, so person B's advocate (advocate_2) loses",
    },
    who_loses_key=lambda d: d["s2"],
    verdict_note="ACTION_A / ACTION_B scored by token equality; UNRESOLVED and NOVERDICT are non-codable",
)


def _games_codable(v: str) -> bool:
    return bool(v) and v != "NOVERDICT"


def _games_correct(r: dict) -> Optional[bool]:
    if not _games_codable(r["verdict"]):
        return None
    c = r.get("correct", "")
    return None if c == "" else c == "1"


GAMES_VALUE = Instrument(
    name="games_value_region",
    codable=_games_codable,
    correct=_games_correct,
    strata={
        "dilemma_both_prefer": ("DILEMMA_BOTH_PREFER games, the cell both players prefer is gold; "
                                "no seat loses under it (16.13 B1)",
                                lambda d: d["gold_type"] == "DILEMMA_BOTH_PREFER"),
        "assurance": ("ASSURANCE games, the payoff-dominant equilibrium is gold; "
                      "no seat loses under it",
                      lambda d: d["gold_type"] == "ASSURANCE"),
    },
    who_loses={
        "DILEMMA_BOTH_PREFER": "both players prefer the gold cell, no loser",
        "ASSURANCE": "both players prefer the gold equilibrium, no loser",
    },
    who_loses_key=lambda d: d["gold_type"],
    verdict_note="PLAN_xy tokens (and NO_RECOMMENDATION in the non-binding regime) scored by the rows CSV `correct` column",
)

GAMES_COORD = Instrument(
    name="games_coordination",
    codable=_games_codable,
    correct=_games_correct,
    strata={
        "one_loser": ("the group recommended a pure equilibrium, which favours one player (16.13d)",
                      lambda d: d["s2"].startswith("PLAN_")),
        "no_recommendation": ("the group declined to recommend, no loser",
                              lambda d: d["s2"] == "NO_RECOMMENDATION"),
    },
    who_loses={
        "PLAN": "an equilibrium favouring one player was recommended, the other player's advocate loses",
        "NO_RECOMMENDATION": "no recommendation, no loser",
    },
    who_loses_key=lambda d: "PLAN" if d["s2"].startswith("PLAN_") else d["s2"],
    verdict_note=("PLAN_xy or NO_RECOMMENDATION; the rows CSV `correct` column scores against the binding "
                  "classification's gold (NO_UNIQUE_PLAN), NOT the 16.13d COORD readout, which lives in "
                  "game_coordination_analysis.json; the sensor block here is therefore not a 16.13d reading"),
)

GAMES_EXTENT = Instrument(
    name="games_extent_map",
    codable=_games_codable,
    correct=_games_correct,
    strata={
        "value": ("value region (DILEMMA_BOTH_PREFER, ASSURANCE)", lambda d: d["region"] == "value"),
        "neutral": ("neutral region", lambda d: d["region"] == "neutral"),
        "cannot_assist": ("cannot-assist region (no unique plan)", lambda d: d["region"] == "cannot_assist"),
    },
    who_loses={},
    who_loses_key=lambda d: d["gold_type"],
    verdict_note="PLAN_xy or NO_UNIQUE_PLAN scored by the rows CSV `correct` column",
)


# ---------------------------------------------------------------------------
# Communities
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Community:
    key: str
    tag: str
    label: str
    instrument: Instrument
    seats: tuple                    # role ids in (advocate_1, advocate_2, third) order
    model: str
    requested: bool = True
    cell: Optional[tuple] = None    # (roles, edges) for the 16.10 factorial
    family: str = "aita"
    extra_fields: tuple = ()


AITA_SEATS = ("writer_advocate", "counterparty", "neutral_adjudicator")
GROK = "grok-4-1-fast-reasoning"

COMMUNITIES: tuple = (
    Community("grok_aita", "cg_deliberation", "grok AITA, embodied roles, edges on", AITA, AITA_SEATS, GROK,
              cell=("embodied", "on")),
    Community("grok_aita_noedge", "cg_deliberation_noedge", "grok AITA, embodied roles, edges off", AITA,
              tuple(s + "_noedge" for s in AITA_SEATS), GROK, cell=("embodied", "off")),
    Community("grok_aita_identical", "cg_deliberation_identical", "grok AITA, identical readers, edges on", AITA,
              ("reader_alpha", "reader_beta", "reader_gamma"), GROK, cell=("identical", "on")),
    Community("grok_aita_identical_noedge", "cg_deliberation_identical_noedge",
              "grok AITA, identical readers, edges off", AITA,
              ("reader_alpha_noedge", "reader_beta_noedge", "reader_gamma_noedge"), GROK,
              cell=("identical", "off")),
    Community("nano_aita", "cg_deliberation_nano", "gpt-5.4-nano AITA, embodied roles, edges on", AITA,
              AITA_SEATS, "gpt-5.4-nano"),
    Community("haiku_aita", "cg_deliberation_haiku_fixed", "claude-haiku-4-5 AITA, embodied roles, edges on",
              AITA, AITA_SEATS, "claude-haiku-4-5"),
    Community("haiku_aita_249", "cg_deliberation_haiku_249",
              "claude-haiku-4-5 AITA, embodied roles, edges on, full 249-item panel (16.23)",
              AITA, AITA_SEATS, "claude-haiku-4-5"),
    Community("llama_aita", "cg_deliberation_llama3370binstruct",
              "Llama-3.3-70B-Instruct AITA, embodied roles, edges on, 225-item filtered panel (16.24)",
              AITA, AITA_SEATS, "Llama-3.3-70B-Instruct"),
    Community("deepseek_aita", "cg_deliberation_deepseekv4pro",
              "DeepSeek-V4-Pro AITA, embodied roles, edges on, 210-item filtered panel (16.27)",
              AITA, AITA_SEATS, "DeepSeek-V4-Pro"),
    Community("mistral_aita", "cg_deliberation_mistrallarge32",
              "Mistral-Large-3-2 AITA, embodied roles, edges on, 217-item filtered panel (16.26)",
              AITA, AITA_SEATS, "Mistral-Large-3-2"),
    Community("grok_dilemmas", "cg_dilemma_stage1", "grok Scruples Dilemmas, Stage 1", DILEMMAS,
              ("person_a_advocate", "person_b_advocate", "neutral_adjudicator"), GROK, family="dilemmas"),
    Community("grok_games_binding_neutral", "cg_game_neutral", "grok games, binding B1, neutral reader",
              GAMES_VALUE, ("player_r_advocate", "player_c_advocate", "neutral_reader"), GROK,
              family="games", extra_fields=("gold_type", "region")),
    Community("grok_games_binding_planner", "cg_game_planner", "grok games, binding B1, plan mediator",
              GAMES_VALUE, ("player_r_advocate", "player_c_advocate", "plan_mediator"), GROK,
              family="games", extra_fields=("gold_type", "region")),
    Community("grok_games_nonbinding_neutral", "cgnb_game_neutral", "grok games, non-binding C1, neutral reader",
              GAMES_VALUE, ("player_r_advocate_nb", "player_c_advocate_nb", "neutral_reader_nb"), GROK,
              family="games", extra_fields=("gold_type", "region")),
    Community("grok_games_nonbinding_planner", "cgnb_game_planner", "grok games, non-binding C1, plan mediator",
              GAMES_VALUE, ("player_r_advocate_nb", "player_c_advocate_nb", "plan_mediator_nb"), GROK,
              family="games", extra_fields=("gold_type", "region")),
    Community("grok_games_coordination_neutral", "cgnb_game_neutral_d",
              "grok games, non-binding 16.13d coordination, neutral reader", GAMES_COORD,
              ("player_r_advocate_nb", "player_c_advocate_nb", "neutral_reader_nb"), GROK, requested=False,
              family="games", extra_fields=("gold_type", "region")),
    Community("grok_games_coordination_planner", "cgnb_game_planner_d",
              "grok games, non-binding 16.13d coordination, plan mediator", GAMES_COORD,
              ("player_r_advocate_nb", "player_c_advocate_nb", "plan_mediator_nb"), GROK, requested=False,
              family="games", extra_fields=("gold_type", "region")),
    Community("grok_games_extent_neutral", "cg_game_neutral_b2", "grok games, binding B2 extent map, neutral reader",
              GAMES_EXTENT, ("player_r_advocate", "player_c_advocate", "neutral_reader"), GROK, requested=False,
              family="games", extra_fields=("gold_type", "region")),
    Community("grok_games_extent_planner", "cg_game_planner_b2", "grok games, binding B2 extent map, plan mediator",
              GAMES_EXTENT, ("player_r_advocate", "player_c_advocate", "plan_mediator"), GROK, requested=False,
              family="games", extra_fields=("gold_type", "region")),
)
COMMUNITY_BY_KEY = {c.key: c for c in COMMUNITIES}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def read_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def build_debates(rows: Sequence[dict], votes: Sequence[dict], com: Community) -> list[dict]:
    """One record per debate. Codability is a field, never a filter, so the
    three populations are cut from one list. Seat objections come from the
    votes CSV `objected_r3` bit, keyed by seat POSITION (advocate_1,
    advocate_2, third) so that every community goes through identical code."""
    inst = com.instrument
    pos_of = {seat: pos for seat, pos in zip(com.seats, SEAT_POSITIONS)}
    debates: dict[tuple, dict] = {}
    for r in rows:
        if com.model and r.get("model") and r["model"] != com.model:
            continue
        key = (r["arm"], r["item_id"], str(r["sample_idx"]))
        s1, s2 = r.get("synthesis_verdict", ""), r.get("verdict", "")
        d = {
            "item": r["item_id"], "arm": r["arm"], "sample_idx": int(r["sample_idx"]),
            "gold": r["gold_verdict"], "s1": s1, "s2": s2,
            "s1_codable": inst.codable(s1), "s2_codable": inst.codable(s2),
            "s2_ok": inst.correct(r),
            "n_objectors": int(r.get("n_objectors") or 0),
            "n_r3_unparsed": int(r.get("n_r3_unparsed") or 0),
            "obj": {}, "seats_seen": set(),
        }
        for fld in com.extra_fields:
            d[fld] = r.get(fld, "")
        debates[key] = d
    for v in votes:
        d = debates.get((v["arm"], v["item_id"], str(v["sample_idx"])))
        if d is None:
            continue
        pos = pos_of.get(v["role_id"])
        if pos is None:
            continue
        d["seats_seen"].add(pos)
        d["obj"][pos] = v.get("objected_r3") == "1"
    for d in debates.values():
        d["seats_complete"] = all(p in d["seats_seen"] for p in SEAT_POSITIONS)
        d["n_objectors_votes"] = sum(1 for p in SEAT_POSITIONS if d["obj"].get(p))
        d["seats_seen"] = sorted(d["seats_seen"])
    return sorted(debates.values(), key=lambda d: (d["item"], d["arm"], d["sample_idx"]))


def population(debates: Sequence[dict], which: str) -> list[dict]:
    if which == "raw":
        return list(debates)
    if which == "s2":
        return [d for d in debates if d["s2_codable"]]
    if which == "s1s2":
        return [d for d in debates if d["s2_codable"] and d["s1_codable"]]
    raise ValueError(which)


# ---------------------------------------------------------------------------
# Item-clustered bootstrap on per-item count vectors
# ---------------------------------------------------------------------------

def phi_from_table(a: int, b: int, c: int, d: int) -> Optional[float]:
    """phi for the 2x2 [[a, b], [c, d]] = [[both, x only], [y only, neither]].
    None when a margin is zero."""
    den = (a + b) * (c + d) * (a + c) * (b + d)
    if den <= 0:
        return None
    return (a * d - b * c) / math.sqrt(den)


def draw_indices(n_items: int, draws: int, seed: int) -> np.ndarray:
    """The resampling sequence of analyze_unembodied_ablation._boot, draw for
    draw. One randrange per item per draw from random.Random(seed)."""
    rng = random.Random(seed)
    idx = np.empty((draws, n_items), dtype=np.int64)
    for k in range(draws):
        for j in range(n_items):
            idx[k, j] = rng.randrange(n_items)
    return idx


class Frame:
    """Per-item count vectors over a fixed set of columns, for one population.
    Every statistic is a function of column sums, so a resample is a sum of
    rows. `ids` are the sorted item ids of the FULL population (the frame);
    an item with no debate in a stratum contributes a zero row there."""

    def __init__(self, ids: Sequence[str], columns: Sequence[str]):
        self.ids = list(ids)
        self.columns = list(columns)
        self.col = {c: i for i, c in enumerate(self.columns)}
        self.M = np.zeros((len(self.ids), len(self.columns)), dtype=np.int64)
        self.row = {i: k for k, i in enumerate(self.ids)}
        self._idx: Optional[np.ndarray] = None

    def add(self, item: str, column: str, value: int = 1) -> None:
        self.M[self.row[item], self.col[column]] += value

    def totals(self) -> dict:
        s = self.M.sum(axis=0)
        return {c: int(s[i]) for c, i in self.col.items()}

    def boot(self, stat: Callable[[dict], Optional[float]], *, draws: int, seed: int,
             alpha: float = 0.05, idx: Optional[np.ndarray] = None) -> dict:
        """Point on the full frame plus the percentile interval over `draws`
        resamples of the items. `stat` maps a dict of column sums to a float
        or None; None draws are dropped (the _boot convention)."""
        point = stat(self.totals())
        if idx is None:
            idx = draw_indices(len(self.ids), draws, seed)
        sums = self.M[idx].sum(axis=1)          # draws x columns
        vals = []
        for k in range(sums.shape[0]):
            v = stat({c: int(sums[k, i]) for c, i in self.col.items()})
            if v is not None:
                vals.append(v)
        vals.sort()
        if not vals:
            return {"point": point, "lo": None, "hi": None}
        lo = vals[int(alpha / 2 * len(vals))]
        hi = vals[min(int((1 - alpha / 2) * len(vals)), len(vals) - 1)]
        return {"point": point, "lo": lo, "hi": hi}


def _rate(num: str, den: str):
    def stat(s):
        return (s[num] / s[den]) if s[den] else None
    return stat


def _phi(both: str, a_only: str, b_only: str, n: str):
    def stat(s):
        a, b, c = s[both], s[a_only], s[b_only]
        return phi_from_table(a, b, c, s[n] - a - b - c)
    return stat


def _lift(wf: str, f: str, wu: str, u: str):
    def stat(s):
        if not s[f] or not s[u]:
            return None
        return s[wf] / s[f] - s[wu] / s[u]
    return stat


# ---------------------------------------------------------------------------
# Objection graph on one population, strata inside the resample
# ---------------------------------------------------------------------------

def graph_columns(strata: Sequence[str]) -> list[str]:
    cols = []
    for g in strata:
        cols += [f"{g}:n", f"{g}:fired"]
        cols += [f"{g}:obj:{p}" for p in SEAT_POSITIONS]
        for a, b in PAIRS:
            cols += [f"{g}:both:{a}|{b}", f"{g}:only:{a}|{b}", f"{g}:only:{b}|{a}"]
    return cols


def fill_graph_frame(fr: Frame, pop: Sequence[dict], strata: dict) -> None:
    for d in pop:
        for g, pred in strata.items():
            if not pred(d):
                continue
            fr.add(d["item"], f"{g}:n")
            if d["n_objectors"] >= 2:
                fr.add(d["item"], f"{g}:fired")
            o = d["obj"]
            for p in SEAT_POSITIONS:
                if o.get(p):
                    fr.add(d["item"], f"{g}:obj:{p}")
            for a, b in PAIRS:
                oa, ob = bool(o.get(a)), bool(o.get(b))
                if oa and ob:
                    fr.add(d["item"], f"{g}:both:{a}|{b}")
                elif oa:
                    fr.add(d["item"], f"{g}:only:{a}|{b}")
                elif ob:
                    fr.add(d["item"], f"{g}:only:{b}|{a}")


def graph_block(fr: Frame, g: str, pop: Sequence[dict], pred, com: Community, *,
                draws: int, seed: int, idx: np.ndarray) -> dict:
    tot = fr.totals()
    n = tot[f"{g}:n"]
    members = [d for d in pop if pred(d)]
    seat_names = dict(zip(SEAT_POSITIONS, com.seats))
    nodes = {}
    for p in SEAT_POSITIONS:
        ci = fr.boot(_rate(f"{g}:obj:{p}", f"{g}:n"), draws=draws, seed=seed, idx=idx)
        nodes[p] = {"seat": seat_names[p], "n_objected": tot[f"{g}:obj:{p}"], "n": n,
                    "objection_rate": ci}
    edges = {}
    for a, b in PAIRS:
        both, ao, bo = tot[f"{g}:both:{a}|{b}"], tot[f"{g}:only:{a}|{b}"], tot[f"{g}:only:{b}|{a}"]
        ci = fr.boot(_phi(f"{g}:both:{a}|{b}", f"{g}:only:{a}|{b}", f"{g}:only:{b}|{a}", f"{g}:n"),
                     draws=draws, seed=seed, idx=idx)
        edges[f"{a}|{b}"] = {"seats": [seat_names[a], seat_names[b]], "phi": ci,
                             "table": {"both": both, f"{a}_only": ao, f"{b}_only": bo,
                                       "neither": n - both - ao - bo},
                             "p_both_object": (both / n) if n else None}
    fire = fr.boot(_rate(f"{g}:fired", f"{g}:n"), draws=draws, seed=seed, idx=idx)
    counts = Counter(d["n_objectors"] for d in members)
    return {
        "n": n, "n_items_kept": len({d["item"] for d in members}), "n_items_frame": len(fr.ids),
        "nodes": nodes, "edges": edges,
        "fire_rate": {**fire, "counter": "stake-blind n_objectors >= 2", "n_fired": tot[f"{g}:fired"]},
        "objector_counts": {str(k): counts.get(k, 0) for k in range(4)},
        "p_exactly_one": (counts.get(1, 0) / n) if n else None,
    }


def objection_graphs(pop: Sequence[dict], com: Community, *, draws: int, seed: int) -> dict:
    """Graphs on `pop` (the frame) for the whole population and for each of
    the instrument's strata, restriction applied inside each resample."""
    strata = {"all": lambda d: True, **{k: v[1] for k, v in com.instrument.strata.items()}}
    ids = sorted({d["item"] for d in pop})
    fr = Frame(ids, graph_columns(list(strata)))
    fill_graph_frame(fr, pop, strata)
    idx = draw_indices(len(ids), draws, seed)
    out = {}
    for g, pred in strata.items():
        blk = graph_block(fr, g, pop, pred, com, draws=draws, seed=seed, idx=idx)
        blk["description"] = ("every debate of the population" if g == "all"
                              else com.instrument.strata[g][0])
        out[g] = blk
    return out


# ---------------------------------------------------------------------------
# Who loses: which seat objects under each verdict type
# ---------------------------------------------------------------------------

def who_loses(pop: Sequence[dict], com: Community, *, draws: int, seed: int) -> dict:
    inst = com.instrument
    ids = sorted({d["item"] for d in pop})
    groups = sorted({inst.who_loses_key(d) for d in pop})
    cols = []
    for g in groups:
        cols += [f"{g}:n"] + [f"{g}:obj:{p}" for p in SEAT_POSITIONS]
    fr = Frame(ids, cols)
    for d in pop:
        g = inst.who_loses_key(d)
        fr.add(d["item"], f"{g}:n")
        for p in SEAT_POSITIONS:
            if d["obj"].get(p):
                fr.add(d["item"], f"{g}:obj:{p}")
    idx = draw_indices(len(ids), draws, seed)
    tot = fr.totals()
    seat_names = dict(zip(SEAT_POSITIONS, com.seats))
    out = {}
    for g in groups:
        n = tot[f"{g}:n"]
        rates = {p: {"seat": seat_names[p], "n_objected": tot[f"{g}:obj:{p}"],
                     **fr.boot(_rate(f"{g}:obj:{p}", f"{g}:n"), draws=draws, seed=seed, idx=idx)}
                 for p in SEAT_POSITIONS}
        top = max(SEAT_POSITIONS, key=lambda p: rates[p]["point"] if rates[p]["point"] is not None else -1)
        out[g] = {"n": n, "who_loses": inst.who_loses.get(g, "not defined for this instrument"),
                  "objection_rate": rates,
                  "most_frequent_objector": {"position": top, "seat": seat_names[top],
                                             "rate": rates[top]["point"]}}
    return out


# ---------------------------------------------------------------------------
# Sensor: the stake-blind counter and its error lift inside each stratum
# ---------------------------------------------------------------------------

def sensor_block(pop: Sequence[dict], com: Community, *, draws: int, seed: int) -> dict:
    strata = {"all": lambda d: True, **{k: v[1] for k, v in com.instrument.strata.items()}}
    scored = [d for d in pop if d["s2_ok"] is not None]
    ids = sorted({d["item"] for d in pop})
    cols = []
    for g in strata:
        cols += [f"{g}:f", f"{g}:u", f"{g}:wf", f"{g}:wu"]
    fr = Frame(ids, cols)
    for d in scored:
        fired = d["n_objectors"] >= 2
        wrong = not d["s2_ok"]
        for g, pred in strata.items():
            if not pred(d):
                continue
            fr.add(d["item"], f"{g}:f" if fired else f"{g}:u")
            if wrong:
                fr.add(d["item"], f"{g}:wf" if fired else f"{g}:wu")
    idx = draw_indices(len(ids), draws, seed)
    tot = fr.totals()
    out = {"n_scored": len(scored), "n_frame": len(pop), "n_items_frame": len(ids), "strata": {}}
    for g in strata:
        f, u, wf, wu = tot[f"{g}:f"], tot[f"{g}:u"], tot[f"{g}:wf"], tot[f"{g}:wu"]
        pwf = (wf / f) if f else None
        pwu = (wu / u) if u else None
        lift = fr.boot(_lift(f"{g}:wf", f"{g}:f", f"{g}:wu", f"{g}:u"), draws=draws, seed=seed, idx=idx)
        out["strata"][g] = {
            "n": f + u, "n_fired": f, "n_unfired": u,
            "fire_rate": (f / (f + u)) if (f + u) else None,
            "p_wrong_fired": pwf, "p_wrong_unfired": pwu,
            "ratio": (pwf / pwu) if (pwf is not None and pwu) else None,
            "lift": lift,
            "ci_excludes_zero": (lift["lo"] is not None and (lift["lo"] > 0 or lift["hi"] < 0)),
            "p_wrong_overall": ((wf + wu) / (f + u)) if (f + u) else None,
        }
    return out


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def run_guards(debates: Sequence[dict]) -> dict:
    n_raw = len(debates)
    s2 = population(debates, "s2")
    n_cod = len(s2)
    share = (n_cod / n_raw) if n_raw else 0.0
    mean_unparsed = (sum(d["n_r3_unparsed"] for d in debates) / n_raw) if n_raw else None
    seat_cov = (sum(1 for d in debates if d["seats_complete"]) / n_raw) if n_raw else 0.0
    consistent = (sum(1 for d in debates if d["n_objectors"] == d["n_objectors_votes"]) / n_raw) if n_raw else 0.0
    checks = {
        "seat_vote_coverage": {"value": seat_cov, "limit": MIN_SEAT_COVERAGE, "ok": seat_cov >= MIN_SEAT_COVERAGE,
                               "structural": True},
        "n_objectors_matches_per_seat_sum": {"value": consistent, "limit": MIN_OBJECTOR_CONSISTENCY,
                                             "ok": consistent >= MIN_OBJECTOR_CONSISTENCY, "structural": True},
        "mean_n_r3_unparsed": {"value": mean_unparsed, "limit": MAX_MEAN_R3_UNPARSED,
                               "ok": mean_unparsed is not None and mean_unparsed <= MAX_MEAN_R3_UNPARSED,
                               "structural": True},
        "s2_codable_share": {"value": share, "limit": MIN_CODABLE_SHARE, "ok": share >= MIN_CODABLE_SHARE,
                             "structural": False},
    }
    structural_ok = all(c["ok"] for c in checks.values() if c["structural"])
    status = "READ" if all(c["ok"] for c in checks.values()) else ("READ_WITH_CAUTION" if structural_ok else "UNREAD")
    return {"n_raw": n_raw, "n_s2_codable": n_cod, "checks": checks,
            "structural_passed": structural_ok, "status": status}


# ---------------------------------------------------------------------------
# One community
# ---------------------------------------------------------------------------

def raw_block(debates: Sequence[dict], com: Community) -> dict:
    """The 16.10 convention, every debate and every vote, points only."""
    n = len(debates)
    seat_names = dict(zip(SEAT_POSITIONS, com.seats))
    complete = [d for d in debates if d["seats_complete"]]
    rates = {}
    for p in SEAT_POSITIONS:
        # per-seat marginal over every vote the seat cast (the 16.10 per_seat_objection_rates convention)
        cast = [d for d in debates if p in d["obj"]]
        rates[p] = {"seat": seat_names[p], "n_votes": len(cast),
                    "objection_rate": (sum(1 for d in cast if d["obj"][p]) / len(cast)) if cast else None}
    counts = Counter(d["n_objectors"] for d in debates)
    noncod = [d for d in debates if not d["s2_codable"]]
    n_fired_noncod = sum(1 for d in noncod if d["n_objectors"] >= 2)
    return {
        "n_debates": n, "n_items": len({d["item"] for d in debates}), "n_seats_complete": len(complete),
        "n_s2_noncodable": len(noncod), "n_fired_among_noncodable": n_fired_noncod,
        "fire_rate_noncodable": (n_fired_noncod / len(noncod)) if noncod else None,
        "objection_rate": rates,
        "pooled_objection_rate": (sum(r["objection_rate"] * r["n_votes"] for r in rates.values()
                                      if r["objection_rate"] is not None)
                                  / sum(r["n_votes"] for r in rates.values())) if n else None,
        "fire_rate": ((counts.get(2, 0) + counts.get(3, 0)) / n) if n else None,
        "n_fired": counts.get(2, 0) + counts.get(3, 0),
        "objector_counts": {str(k): counts.get(k, 0) for k in range(4)},
        "p_exactly_one": (counts.get(1, 0) / n) if n else None,
        "raw_verdict_counts": dict(sorted(Counter(d["s2"] for d in debates).items())),
    }


def analyse_community(com: Community, debates: Sequence[dict], *, draws: int, seed: int) -> dict:
    guards = run_guards(debates)
    out: dict = {
        "key": com.key, "tag": com.tag, "label": com.label, "model": com.model,
        "instrument": com.instrument.name, "family": com.family, "requested": com.requested,
        "cell_16_10": ({"roles": com.cell[0], "edges": com.cell[1]} if com.cell else None),
        "seats": dict(zip(SEAT_POSITIONS, com.seats)),
        "verdict_note": com.instrument.verdict_note,
        "guards": guards, "status": guards["status"],
        "raw": raw_block(debates, com),
    }
    if guards["status"] == "UNREAD":
        out["reason"] = "a structural guard failed; no graph is written"
        return out
    s2 = population(debates, "s2")
    s1s2 = population(debates, "s1s2")
    out["populations"] = {
        "s2_codable": {
            "description": "final integrated verdict (S2) codable; the 16.12 population",
            "n": len(s2), "n_items": len({d["item"] for d in s2}),
            "verdict_counts": dict(sorted(Counter(d["s2"] for d in s2).items())),
            "graphs": objection_graphs(s2, com, draws=draws, seed=seed),
            "who_loses": who_loses(s2, com, draws=draws, seed=seed + 1),
        },
        "s1_s2_codable": {
            "description": "S1 synthesis and S2 final verdict both codable; the 16.15 population",
            "n": len(s1s2), "n_items": len({d["item"] for d in s1s2}),
            "sensor": sensor_block(s1s2, com, draws=draws, seed=seed),
        },
    }
    if guards["status"] == "READ_WITH_CAUTION":
        out["caution"] = (f"S2-codable share {guards['checks']['s2_codable_share']['value']:.3f} is below "
                          f"{MIN_CODABLE_SHARE}; the objection graph does not depend on codability, "
                          "the strata and the sensor lift do and are read with that caution")
    return out


def load_community(com: Community, out_dir: Path = OUT_DIR) -> Optional[list[dict]]:
    rows_path, votes_path = out_dir / f"{com.tag}_rows.csv", out_dir / f"{com.tag}_votes.csv"
    if not rows_path.exists() or not votes_path.exists():
        return None
    return build_debates(read_csv(rows_path), read_csv(votes_path), com)


# ---------------------------------------------------------------------------
# The deployed protocol as a graph (structure from the runner, numbers from data)
# ---------------------------------------------------------------------------

def protocol_block(res: dict) -> dict:
    """Nodes and edges of the registered seven-stage protocol, with the three
    annotations the figure carries read from this artefact (fire rate, the
    one-loser ratio) and from router_decomposition.json (the routing gain)."""
    g = res["communities"].get("grok_aita")
    fire = g["raw"]["fire_rate"] if g and g["status"] != "UNREAD" else None
    sensor = (g["populations"]["s1_s2_codable"]["sensor"]["strata"] if g and g["status"] != "UNREAD" else {})
    # which seats make up the stake-blind counter's firings within one-loser verdicts (the
    # 2026-09-13 review: the flag is a three-seat count, not the W, C pair alone)
    split = None
    if g and g["status"] != "UNREAD":
        ol = g["populations"]["s2_codable"]["graphs"].get("one_loser")
        if ol and ol.get("n"):
            both = ol["edges"]["advocate_1|advocate_2"]["table"]["both"]
            n_fired = ol["fire_rate"]["n_fired"]
            split = {"population": "s2_codable, one-loser verdicts (YTA or NTA)", "n": ol["n"],
                     "n_fired": n_fired, "both_advocates_object": both,
                     "neutral_with_one_advocate": n_fired - both,
                     "neutral_objected": ol["nodes"]["third"]["n_objected"],
                     "objector_counts": ol.get("objector_counts"),
                     "counter": ol["fire_rate"].get("counter")}
    routing = None
    if ROUTER_PATH.exists():
        rd = json.loads(ROUTER_PATH.read_text())
        hk = rd["composed_haiku"]
        sn = rd["composed_sonnet"]["block"]
        routing = {
            "source": ROUTER_PATH.name,
            "rule": "S2 unless the counter fired, then the cached judge's standard majority-of-3",
            "haiku_judge": {"collective_alone": hk["routers"]["collective_alone"]["composed"]["acc"],
                            "counter_routed": hk["routers"]["counter"]["composed"]["acc"],
                            "routed": hk["routers"]["counter"]["n_fire"],
                            "gain": hk["paired_deltas"]["counter_minus_collective_alone"]},
            "sonnet_judge": {"collective_alone": sn["routers"]["collective_alone"]["composed"]["acc"],
                             "counter_routed": sn["routers"]["counter"]["composed"]["acc"],
                             "routed": sn["routers"]["counter"]["n_fire"],
                             "gain": sn["paired_deltas"]["counter_minus_collective_alone"]},
        }
    return {
        "nodes": {
            "W": {"seat": "writer_advocate", "kind": "advocate", "stake": "the writer is not at fault"},
            "C": {"seat": "counterparty", "kind": "advocate", "stake": "the writer is at fault"},
            "N": {"seat": "neutral_adjudicator", "kind": "third seat", "stake": "none"},
            "M": {"seat": "moderator", "kind": "moderator", "stake": "none"},
            "J": {"seat": "judge", "kind": "actuator", "stake": "none"},
        },
        "stages": ["r0 position", "r1 rebuttal", "r2 final position", "S1 synthesis",
                   "r3 label", "S2 integration", "r4 vote"],
        "edges": [
            {"from": "W", "to": "C", "stage": "r1, r2", "kind": "read", "directed": True},
            {"from": "W", "to": "N", "stage": "r1, r2", "kind": "read", "directed": True},
            {"from": "C", "to": "W", "stage": "r1, r2", "kind": "read", "directed": True},
            {"from": "C", "to": "N", "stage": "r1, r2", "kind": "read", "directed": True},
            {"from": "N", "to": "W", "stage": "r1, r2", "kind": "read", "directed": True},
            {"from": "N", "to": "C", "stage": "r1, r2", "kind": "read", "directed": True},
            {"from": "W", "to": "M", "stage": "S1", "kind": "synthesis", "directed": True},
            {"from": "C", "to": "M", "stage": "S1", "kind": "synthesis", "directed": True},
            {"from": "N", "to": "M", "stage": "S1", "kind": "synthesis", "directed": True},
            {"from": "M", "to": "W", "stage": "r3", "kind": "label", "directed": True},
            {"from": "M", "to": "C", "stage": "r3", "kind": "label", "directed": True},
            {"from": "M", "to": "N", "stage": "r3", "kind": "label", "directed": True},
            {"from": "W", "to": "M", "stage": "r4", "kind": "vote", "directed": True},
            {"from": "C", "to": "M", "stage": "r4", "kind": "vote", "directed": True},
            {"from": "N", "to": "M", "stage": "r4", "kind": "vote", "directed": True},
            {"seats": ["W", "C", "N"], "stage": "r3", "kind": "flag", "directed": False, "hyperedge": True,
             "counter": "stake-blind n_objectors >= 2 over the three seats' r3 labels",
             "note": ("the three-seat objection set the stake-blind counter reads. the counter is not the W, C "
                      "pair alone: within one-loser verdicts the neutral seat's objection sits inside the firings "
                      "that both advocates do not make (annotations.one_loser_firings_split), and the "
                      "anti-correlated W, C pair is the edge whose against-interest crossings carry the signal")},
            {"from": "M", "to": "J", "stage": "after r4", "kind": "route", "directed": True,
             "note": "the debate is routed to the judge when two or more seats objected"},
        ],
        "edges_removed_in_noedge_cell": "the six r1, r2 read edges among the seats",
        "annotations": {
            "fire_rate_raw": fire,
            "one_loser_firings_split": split,
            "n_sensor_population": (g["populations"]["s1_s2_codable"]["n"] if sensor else None),
            "one_loser_ratio": (sensor.get("one_loser", {}).get("ratio") if sensor else None),
            "one_loser_lift": (sensor.get("one_loser", {}).get("lift") if sensor else None),
            "routing_gain": routing,
            "routing_gain_over_solo_16_11a": RECORD_16_11A,
        },
    }


# The 16.11a per-cell comparison (routed system minus the grok solo majority-of-3)
# has no JSON artefact and no reproducing script in scripts/; its numbers were
# printed to the console and entered in the record. They are carried here as a
# RECORD QUOTE with provenance so that a figure can annotate them and --check can
# assert equality against this artefact, never as a computed value.
RECORD_16_11A = {
    "kind": "record quote, not computed by this module",
    "source": "Guidance_Documents/prereg_embodiment_community.md, 16.11a RESULTS (run 2026-09-12)",
    "unit": "418 (arm, item) cells on 210 items; item-clustered 95% CI",
    "base": {"system": "grok solo, majority-of-3", "accuracy": 0.9115},
    "routed_embodied": {"system": "grok solo -> sonnet on the embodied flag", "coverage": 0.127,
                        "accuracy": 0.9354, "delta_vs_solo": 0.0239, "lo": 0.0024, "hi": 0.0452},
    "routed_union": {"system": "grok solo -> sonnet on the union sensor", "coverage": 0.175,
                     "accuracy": 0.9498, "delta_vs_solo": 0.0383, "lo": 0.0121, "hi": 0.0646},
    "sonnet_everywhere": {"system": "sonnet solo, majority-of-3, everywhere", "accuracy": 0.9761,
                          "routed_embodied_minus_this": -0.0407, "lo": -0.0692, "hi": -0.0120},
    "reading": ("positive on the first registered criterion (routing beats the base), null on the second "
                "(sonnet everywhere dominates routing); the sensor's value is cost efficiency"),
}


# ---------------------------------------------------------------------------
# Agreement with the record and with the artefacts it cites
# ---------------------------------------------------------------------------

def _agree(got, quoted, tol):
    ok = got is not None and quoted is not None and abs(got - quoted) <= tol
    return {"got": got, "quoted": quoted, "tol": tol, "agrees": bool(ok)}


RECORD_QUOTES = {
    # 16.10 RESULTS, fire rate P(>=2), raw population per cell
    "16_10_fire_rate": {"grok_aita": 0.196, "grok_aita_noedge": 0.129,
                        "grok_aita_identical": 0.005, "grok_aita_identical_noedge": 0.007},
    # 16.16 registration quoting the embodied/on per-seat objection marginals
    "16_10_seat_marginals_grok": {"advocate_1": 0.430, "advocate_2": 0.690, "third": 0.073},
    # 16.12 RESULTS note (S2-codable population)
    "16_12_who_loses": {"YTA": ("advocate_1", 0.992), "NTA": ("advocate_2", 0.988)},
    "16_12_phi": {"one_loser": -0.901, "ESH": -0.191, "NAH": -0.275},
    # 16.15.1 RESULTS (S1 and S2 codable population)
    "16_15_1_one_loser": {"ratio": 9.83, "p_wrong_fired": 0.387, "p_wrong_unfired": 0.039, "lift": 0.348},
    "16_15_1_both_party": {"lift": -0.097, "fire_rate": 0.603},
    "16_15_1_one_loser_fire_rate": 0.072,
    # headline_theory.html section 2.3, pooled edges
    "headline_2_3_pooled_phi": {"advocate_1|advocate_2": -0.72, "advocate_1|third": -0.14, "advocate_2|third": 0.15},
    # 16.13 B1 and 16.13d, objections at r3 and the stake-blind fire rate
    "16_13_b1_objections_any_seat": 0,
    "16_13d_fire_rate": 0.0,
}


def agreement_block(res: dict) -> dict:
    coms = res["communities"]
    out: dict = {"record": {}, "artefacts": {}}

    def com_raw(k):
        return k in coms and coms[k]["status"] != "ABSENT"

    def com_ok(k):
        return k in coms and coms[k]["status"] not in ("ABSENT", "UNREAD")

    # --- record quotes ---
    rec = out["record"]
    rec["16_10_fire_rate_raw"] = {k: _agree(coms[k]["raw"]["fire_rate"], q, RATE_TOL)
                                  for k, q in RECORD_QUOTES["16_10_fire_rate"].items() if com_raw(k)}
    if com_raw("grok_aita"):
        rec["16_10_seat_marginals_grok_raw"] = {
            p: _agree(coms["grok_aita"]["raw"]["objection_rate"][p]["objection_rate"], q, RATE_TOL)
            for p, q in RECORD_QUOTES["16_10_seat_marginals_grok"].items()}
    if com_ok("grok_aita"):
        g = coms["grok_aita"]["populations"]
        wl = g["s2_codable"]["who_loses"]
        rec["16_12_who_loses_s2"] = {
            t: _agree(wl[t]["objection_rate"][p]["point"], q, RATE_TOL)
            for t, (p, q) in RECORD_QUOTES["16_12_who_loses"].items() if t in wl}
        gr = g["s2_codable"]["graphs"]
        rec["16_12_phi_advocates_s2"] = {
            "one_loser": _agree(gr["one_loser"]["edges"]["advocate_1|advocate_2"]["phi"]["point"],
                                RECORD_QUOTES["16_12_phi"]["one_loser"], RATE_TOL)}
        sn = g["s1_s2_codable"]["sensor"]["strata"]
        rec["16_15_1_one_loser_s1s2"] = {
            "ratio": _agree(sn["one_loser"]["ratio"], 9.83, 0.005 + 1e-9),
            "p_wrong_fired": _agree(sn["one_loser"]["p_wrong_fired"], 0.387, RATE_TOL),
            "p_wrong_unfired": _agree(sn["one_loser"]["p_wrong_unfired"], 0.039, RATE_TOL),
            "lift": _agree(sn["one_loser"]["lift"]["point"], 0.348, RATE_TOL),
            "fire_rate": _agree(sn["one_loser"]["fire_rate"], 0.072, RATE_TOL),
        }
        rec["16_15_1_both_party_s1s2"] = {
            "lift": _agree(sn["both_party"]["lift"]["point"], -0.097, RATE_TOL),
            "fire_rate": _agree(sn["both_party"]["fire_rate"], 0.603, RATE_TOL),
        }
        rec["headline_2_3_pooled_phi_s2"] = {
            e: _agree(gr["all"]["edges"][e]["phi"]["point"], q, 0.005 + 1e-9)
            for e, q in RECORD_QUOTES["headline_2_3_pooled_phi"].items()}
    for k in ("grok_games_binding_neutral", "grok_games_binding_planner"):
        if com_raw(k):
            n_obj = sum(coms[k]["raw"]["objection_rate"][p]["objection_rate"] or 0 for p in SEAT_POSITIONS)
            rec.setdefault("16_13_b1_objections_any_seat_raw", {})[k] = _agree(n_obj, 0, EXACT_TOL)
    for k in ("grok_games_coordination_neutral", "grok_games_coordination_planner"):
        if com_raw(k):
            rec.setdefault("16_13d_fire_rate_raw", {})[k] = _agree(coms[k]["raw"]["fire_rate"], 0.0, EXACT_TOL)

    # --- artefacts the record cites ---
    art = out["artefacts"]
    if TOPOLOGY_PATH.exists():
        tp = json.loads(TOPOLOGY_PATH.read_text())
        blk = {}
        for k, c in coms.items():
            if not com_raw(k) or not c.get("cell_16_10"):
                continue
            cell = f"{c['cell_16_10']['roles']}/{c['cell_16_10']['edges']}"
            src = tp["cells"].get(cell)
            if not src:
                continue
            blk[k] = {"fire_rate": _agree(c["raw"]["fire_rate"], src["objection"]["fire_rate"], EXACT_TOL),
                      "per_seat": {p: _agree(c["raw"]["objection_rate"][p]["objection_rate"],
                                             src["objection"]["per_seat"].get(c["seats"][p]), EXACT_TOL)
                                   for p in SEAT_POSITIONS},
                      "n_debates": _agree(c["raw"]["n_debates"], src["n_debates"], 0)}
        art[TOPOLOGY_PATH.name] = blk
    if FLOODING_PATH.exists() and com_ok("grok_aita"):
        fl = json.loads(FLOODING_PATH.read_text())
        if fl["aita"]["status"] == "READ":
            fg = fl["aita"]["populations"]["s2_codable_16_12"]["groups"]
            g = coms["grok_aita"]["populations"]["s2_codable"]
            blk = {}
            for t in ("YTA", "NTA", "ESH", "NAH"):
                blk[t] = {"n": _agree(g["who_loses"][t]["n"], fg[t]["n"], 0),
                          **{p: _agree(g["who_loses"][t]["objection_rate"][p]["point"],
                                       fg[t]["seat_objection_rate"][coms["grok_aita"]["seats"][p]], EXACT_TOL)
                             for p in SEAT_POSITIONS}}
            for s, name in (("one_loser", "one_loser"), ("both_party", "both_party"), ("all", "pooled")):
                e = g["graphs"][s]["edges"]["advocate_1|advocate_2"]["phi"]
                blk[f"phi_{s}"] = {"point": _agree(e["point"], fg[name]["phi_advocates"]["point"], EXACT_TOL),
                                   "lo": _agree(e["lo"], fg[name]["phi_advocates"]["lo"], EXACT_TOL),
                                   "hi": _agree(e["hi"], fg[name]["phi_advocates"]["hi"], EXACT_TOL),
                                   "n": _agree(g["graphs"][s]["n"], fg[name]["n"], 0)}
                blk[f"fire_{s}"] = _agree(g["graphs"][s]["fire_rate"]["point"], fg[name]["p_two_plus"]["point"],
                                          EXACT_TOL)
            art[FLOODING_PATH.name] = blk
    if ROUTER_PATH.exists() and com_ok("grok_aita"):
        rd = json.loads(ROUTER_PATH.read_text())
        sn = coms["grok_aita"]["populations"]["s1_s2_codable"]["sensor"]["strata"]
        blk = {}
        for s in ("pooled", "one_loser", "both_party"):
            mine = sn["all" if s == "pooled" else s]
            src = rd["counter_lift_by_stratum"][s]
            blk[s] = {"n": _agree(mine["n"], src["n"], 0), "n_fired": _agree(mine["n_fired"], src["n_fired"], 0),
                      "p_wrong_fired": _agree(mine["p_wrong_fired"], src["p_wrong_fired"], EXACT_TOL),
                      "p_wrong_unfired": _agree(mine["p_wrong_unfired"], src["p_wrong_unfired"], EXACT_TOL),
                      "ratio": _agree(mine["ratio"], src["ratio"], EXACT_TOL),
                      "lift": _agree(mine["lift"]["point"], src["lift"], EXACT_TOL),
                      "lo": _agree(mine["lift"]["lo"], src["lo"], EXACT_TOL),
                      "hi": _agree(mine["lift"]["hi"], src["hi"], EXACT_TOL)}
        art[ROUTER_PATH.name] = blk

    def _walk(o):
        if isinstance(o, dict):
            if "agrees" in o and "got" in o:
                yield o["agrees"]
            else:
                for v in o.values():
                    yield from _walk(v)
    flags = list(_walk(out))
    out["n_checks"] = len(flags)
    out["n_agree"] = sum(flags)
    out["all_agree"] = all(flags)
    return out


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyse(loaded: dict, *, draws: int, seed: int) -> dict:
    res: dict = {
        "registration": ("emergent objection graphs over the three seats, every cached community, "
                         "zero spend; who-loses table and the stake-blind sensor by stratum"),
        "draws": draws, "seed": seed,
        "estimator": ("item-clustered percentile bootstrap reproducing analyze_unembodied_ablation._boot "
                      "draw for draw on per-item count vectors; strata restricted inside each draw; "
                      "undefined draws dropped from the interval"),
        "seat_positions": {"advocate_1": "the first advocate (writer's advocate, person A's advocate, "
                                         "row player's advocate, reader alpha)",
                           "advocate_2": "the second advocate (counterparty, person B's advocate, "
                                         "column player's advocate, reader beta)",
                           "third": "the third seat (neutral adjudicator, neutral reader, plan mediator, "
                                    "reader gamma)"},
        "edge_weight": "phi coefficient between the two seats' r3 objection indicators (objected_r3)",
        "node_weight": "the seat's r3 objection rate",
        "populations": {"raw": "every debate (16.10 convention)",
                        "s2_codable": "final verdict codable (16.12 convention); graphs, strata, who-loses",
                        "s1_s2_codable": "synthesis and final verdict codable (16.15 convention); sensor"},
        "seeds": {"graphs": seed, "who_loses": seed + 1, "sensor": seed},
        "communities": {},
    }
    for com in COMMUNITIES:
        debates = loaded.get(com.key)
        if debates is None:
            res["communities"][com.key] = {"key": com.key, "tag": com.tag, "label": com.label,
                                           "requested": com.requested, "status": "ABSENT",
                                           "reason": "rows or votes CSV not on disk"}
            continue
        res["communities"][com.key] = analyse_community(com, debates, draws=draws, seed=seed)
    res["protocol"] = protocol_block(res)
    res["agreement"] = agreement_block(res)
    res["guards_summary"] = {k: c["status"] for k, c in res["communities"].items()}
    res["any_structural_guard_failed"] = any(c["status"] == "UNREAD" for c in res["communities"].values())
    return res


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _f(x, fmt=".3f"):
    return "n/a" if x is None else format(x, fmt)


def _ci(e: dict, fmt=".3f") -> str:
    return f"{_f(e['point'], fmt)} [{_f(e['lo'], fmt)}, {_f(e['hi'], fmt)}]"


def print_report(res: dict) -> None:
    print("\n" + "=" * 96)
    print("EMERGENT OBJECTION GRAPHS, every cached three-seat community")
    print("=" * 96)
    print(f"  estimator item-clustered bootstrap, {res['draws']} draws, seed {res['seed']}")
    for k, c in res["communities"].items():
        print("\n" + "-" * 96)
        print(f"  {k}  [{c.get('tag')}]  {c.get('label')}  status {c['status']}"
              f"{'' if c.get('requested', True) else '  (extra, not requested)'}")
        if c["status"] in ("ABSENT", "UNREAD"):
            print(f"    {c.get('reason')}")
            continue
        r = c["raw"]
        print(f"    raw: {r['n_debates']} debates, {r['n_items']} items; fire P(>=2) {_f(r['fire_rate'])}; "
              f"seat marginals " + ", ".join(f"{v['seat']} {_f(v['objection_rate'])}"
                                             for v in r["objection_rate"].values()))
        if c.get("caution"):
            print(f"    CAUTION {c['caution']}")
        P = c["populations"]
        print(f"    S2-codable {P['s2_codable']['n']} debates; verdicts {P['s2_codable']['verdict_counts']}")
        for g, blk in P["s2_codable"]["graphs"].items():
            nodes = "  ".join(f"{v['seat']} {_ci(v['objection_rate'])}" for v in blk["nodes"].values())
            edges = "  ".join(f"{e.replace('advocate_', 'a').replace('third', 't')} phi {_ci(v['phi'], '+.3f')}"
                              for e, v in blk["edges"].items())
            print(f"      {g:<20} n {blk['n']:>5}  fire {_ci(blk['fire_rate'])}")
            print(f"        nodes  {nodes}")
            print(f"        edges  {edges}")
        wl = P["s2_codable"]["who_loses"]
        if wl:
            print("    who loses (S2-codable), objection rate by seat:")
            for t, blk in wl.items():
                rates = "  ".join(f"{v['seat']} {_ci(v)}" for v in blk["objection_rate"].values())
                print(f"      {t:<20} n {blk['n']:>5}  {rates}")
        sn = P["s1_s2_codable"]["sensor"]
        print(f"    sensor (S1&S2-codable, {sn['n_frame']} debates, {sn['n_scored']} scored):")
        for g, s in sn["strata"].items():
            print(f"      {g:<20} n {s['n']:>5} fired {s['n_fired']:>4} fire {_f(s['fire_rate'])} "
                  f"P(wrong|fired) {_f(s['p_wrong_fired'])} P(wrong|unfired) {_f(s['p_wrong_unfired'])} "
                  f"ratio {_f(s['ratio'], '.2f')}x lift {_ci(s['lift'], '+.3f')}")
        for name, ck in c["guards"]["checks"].items():
            print(f"    guard {name:<36} {_f(ck['value'], '.4f')}  limit {ck['limit']}  "
                  f"{'ok' if ck['ok'] else ('FAIL' if ck['structural'] else 'below limit, caution')}")
    pr = res["protocol"]["annotations"]
    print("\n" + "-" * 96)
    print(f"  protocol annotations: fire rate {_f(pr['fire_rate_raw'])}, one-loser ratio "
          f"{_f(pr['one_loser_ratio'], '.2f')}x, routing gain (counter minus collective) "
          + ("n/a" if not pr["routing_gain"] else
             f"haiku judge {_f(pr['routing_gain']['haiku_judge']['gain']['delta'], '+.3f')}, "
             f"sonnet judge {_f(pr['routing_gain']['sonnet_judge']['gain']['delta'], '+.3f')}"))
    ag = res["agreement"]
    print(f"  agreement with the record and its artefacts: {ag['n_agree']} of {ag['n_checks']} checks agree"
          f" -> {'ALL AGREE' if ag['all_agree'] else 'NOT ALL AGREE'}")
    if not ag["all_agree"]:
        def _walk(o, path=""):
            if isinstance(o, dict):
                if "agrees" in o and "got" in o:
                    if not o["agrees"]:
                        print(f"    DIFFERS {path}: got {o['got']} quoted {o['quoted']}")
                else:
                    for k, v in o.items():
                        _walk(v, f"{path}.{k}" if path else k)
        _walk(ag)
    if res["any_structural_guard_failed"]:
        print(f"  {GUARD_FAILED} on at least one community (see status UNREAD above)")
    else:
        print("  GUARDS PASSED on every community's structural checks")


# ---------------------------------------------------------------------------
# Selftest on synthetic rows
# ---------------------------------------------------------------------------

def _synthetic_aita() -> tuple[list[dict], list[dict]]:
    """Hand case, 20 debates on 20 items. YTA x6 (gold YTA, right): the writer
    objects alone on every debate. NTA x6 (gold YTA on two, wrong): the
    counterparty objects alone. ESH x4 (gold NTA, wrong under the collapse):
    both advocates object on every debate, the neutral on two. NAH x2 (gold
    NTA, right): the four advocate patterns spread so phi is defined. One
    UNRESOLVED S2 and one NTA with a NOVERDICT S1."""
    rows, votes = [], []

    def add(item, s2, gold, obj, s1=None):
        rows.append({"model": GROK, "item_id": item, "arm": "as_asker", "sample_idx": "0",
                     "gold_verdict": gold, "verdict": s2,
                     "synthesis_verdict": s1 if s1 is not None else s2,
                     "n_objectors": str(sum(obj)), "n_r3_unparsed": "0"})
        for s, o in zip(AITA_SEATS, obj):
            votes.append({"model": GROK, "item_id": item, "arm": "as_asker", "sample_idx": "0",
                          "role_id": s, "objected_r3": "1" if o else "0"})

    for i in range(6):
        add(f"yta{i}", "YTA", "YTA", (1, 0, 0))
    for i in range(6):
        add(f"nta{i}", "NTA", "YTA" if i < 2 else "NTA", (0, 1, 0))
    for i in range(4):
        add(f"esh{i}", "ESH", "NTA", (1, 1, 1) if i < 2 else (1, 1, 0))
    add("nah0", "NAH", "NTA", (1, 0, 0))
    add("nah1", "NAH", "NTA", (0, 1, 0))
    add("unres0", "UNRESOLVED", "YTA", (1, 0, 0))
    add("nta_s1nov", "NTA", "NTA", (0, 1, 0), s1="NOVERDICT")
    return rows, votes


def _selftest(draws: int = 300) -> int:
    fails: list[str] = []

    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    check("phi of a perfectly anti-correlated 2x2 is -1, of an independent one 0, None on a zero margin",
          phi_from_table(0, 3, 3, 0) == -1.0 and phi_from_table(1, 1, 1, 1) == 0.0
          and phi_from_table(4, 0, 0, 0) is None)

    # the count-vector bootstrap reproduces analyze_unembodied_ablation._boot draw for draw
    from scripts.analyze_unembodied_ablation import _boot as ref_boot
    rng = random.Random(5)
    by_item = {f"i{k}": [(rng.random() < 0.4, rng.random() < 0.6) for _ in range(rng.randrange(1, 6))]
               for k in range(15)}
    fr = Frame(sorted(by_item), ["n", "x", "both", "xo", "yo"])
    for it, recs in by_item.items():
        for x, y in recs:
            fr.add(it, "n"); fr.add(it, "x", int(x))
            if x and y:
                fr.add(it, "both")
            elif x:
                fr.add(it, "xo")
            elif y:
                fr.add(it, "yo")
    mine_rate = fr.boot(_rate("x", "n"), draws=200, seed=13)
    ref_rate = ref_boot(by_item, lambda rs: sum(r[0] for r in rs) / len(rs) if rs else None, draws=200, seed=13)
    check("rate: point and both interval bounds identical to _boot",
          (mine_rate["point"], mine_rate["lo"], mine_rate["hi"]) == ref_rate)

    def ref_phi(rs):
        a = sum(1 for r in rs if r[0] and r[1]); b = sum(1 for r in rs if r[0] and not r[1])
        c = sum(1 for r in rs if not r[0] and r[1])
        return phi_from_table(a, b, c, len(rs) - a - b - c) if rs else None
    mine_phi = fr.boot(_phi("both", "xo", "yo", "n"), draws=200, seed=13)
    ref_p = ref_boot(by_item, ref_phi, draws=200, seed=13)
    check("phi: point and both interval bounds identical to _boot",
          (mine_phi["point"], mine_phi["lo"], mine_phi["hi"]) == ref_p)

    com = COMMUNITY_BY_KEY["grok_aita"]
    rows, votes = _synthetic_aita()
    debates = build_debates(rows, votes, com)
    check("build_debates keeps every row (20 debates) with complete seats and consistent counters",
          len(debates) == 20 and all(d["seats_complete"] for d in debates)
          and all(d["n_objectors"] == d["n_objectors_votes"] for d in debates))
    check("populations: raw 20, S2-codable 19 (drops UNRESOLVED), S1&S2 18 (drops the NOVERDICT S1)",
          len(population(debates, "raw")) == 20 and len(population(debates, "s2")) == 19
          and len(population(debates, "s1s2")) == 18)
    c = analyse_community(com, debates, draws=draws, seed=13)
    check("guards pass and status READ", c["status"] == "READ")
    G = c["populations"]["s2_codable"]["graphs"]
    check("strata partition the S2-codable population: one_loser 13 + both_party 6 = all 19",
          G["one_loser"]["n"] == 13 and G["both_party"]["n"] == 6 and G["all"]["n"] == 19)
    e = G["one_loser"]["edges"]["advocate_1|advocate_2"]
    check("one-loser advocate edge is perfectly anti-correlated: phi -1 with a degenerate interval, 2x2 0/6/7/0",
          e["phi"]["point"] == -1.0 and e["phi"]["lo"] == -1.0 and e["phi"]["hi"] == -1.0
          and list(e["table"].values()) == [0, 6, 7, 0])
    check("one-loser nodes: writer 6/13, counterparty 7/13, neutral 0; fire rate 0",
          abs(G["one_loser"]["nodes"]["advocate_1"]["objection_rate"]["point"] - 6 / 13) < 1e-12
          and abs(G["one_loser"]["nodes"]["advocate_2"]["objection_rate"]["point"] - 7 / 13) < 1e-12
          and G["one_loser"]["nodes"]["third"]["objection_rate"]["point"] == 0.0
          and G["one_loser"]["fire_rate"]["point"] == 0.0 and G["one_loser"]["fire_rate"]["n_fired"] == 0)
    e2 = G["both_party"]["edges"]["advocate_1|advocate_2"]
    check("both-party advocate edge: 2x2 4/1/1/0, phi defined and negative, fire rate 4/6",
          list(e2["table"].values()) == [4, 1, 1, 0] and e2["phi"]["point"] is not None
          and e2["phi"]["point"] < 0 and abs(G["both_party"]["fire_rate"]["point"] - 4 / 6) < 1e-12)
    check("neutral edges on both-party: neutral objects only with both advocates (2/6 both), phi defined",
          G["both_party"]["edges"]["advocate_1|third"]["table"]["both"] == 2
          and G["both_party"]["edges"]["advocate_1|third"]["phi"]["point"] is not None)
    check("full resampling frame: every stratum resamples all 19 items and keeps only its own",
          G["both_party"]["n_items_frame"] == 19 and G["both_party"]["n_items_kept"] == 6
          and G["one_loser"]["n_items_kept"] == 13)
    W = c["populations"]["s2_codable"]["who_loses"]
    check("who loses: writer objects on 1.000 of YTA, counterparty on 1.000 of NTA, both on 1.000 of ESH",
          W["YTA"]["objection_rate"]["advocate_1"]["point"] == 1.0
          and W["YTA"]["objection_rate"]["advocate_2"]["point"] == 0.0
          and W["NTA"]["objection_rate"]["advocate_2"]["point"] == 1.0
          and W["ESH"]["objection_rate"]["advocate_1"]["point"] == 1.0
          and W["ESH"]["objection_rate"]["advocate_2"]["point"] == 1.0
          and W["ESH"]["objection_rate"]["third"]["point"] == 0.5
          and W["YTA"]["most_frequent_objector"]["position"] == "advocate_1"
          and W["NTA"]["most_frequent_objector"]["position"] == "advocate_2")
    S = c["populations"]["s1_s2_codable"]["sensor"]["strata"]
    check("sensor on S1&S2: one-loser 12 debates, 0 fired, lift undefined; both-party 6, 4 fired",
          S["one_loser"]["n"] == 12 and S["one_loser"]["n_fired"] == 0 and S["one_loser"]["lift"]["point"] is None
          and S["both_party"]["n"] == 6 and S["both_party"]["n_fired"] == 4)
    check("sensor all: 4 fired all wrong (ESH, gold NTA), 14 unfired with 2 wrong, ratio 7.0, lift +6/7",
          S["all"]["n_fired"] == 4 and S["all"]["p_wrong_fired"] == 1.0
          and abs(S["all"]["p_wrong_unfired"] - 2 / 14) < 1e-12
          and abs(S["all"]["ratio"] - 7.0) < 1e-12 and abs(S["all"]["lift"]["point"] - 6 / 7) < 1e-12)
    R = c["raw"]
    check("raw block: 20 debates, fire rate 4/20, writer marginal 12/20, neutral 2/20",
          R["n_debates"] == 20 and abs(R["fire_rate"] - 0.2) < 1e-12
          and abs(R["objection_rate"]["advocate_1"]["objection_rate"] - 0.6) < 1e-12
          and abs(R["objection_rate"]["third"]["objection_rate"] - 0.1) < 1e-12)

    # a structural guard failure leaves the community UNREAD
    bad_votes = [v for v in votes if not (v["role_id"] == "neutral_adjudicator" and v["item_id"].startswith("yta"))]
    c_bad = analyse_community(com, build_debates(rows, bad_votes, com), draws=50, seed=13)
    check("missing seat votes on 6 of 20 debates fails the structural guard and the block is UNREAD",
          c_bad["status"] == "UNREAD" and "populations" not in c_bad)

    # a codability shortfall alone is READ_WITH_CAUTION
    rows_c = [dict(r) for r in rows]
    for r in rows_c[:3]:
        r["verdict"] = "UNRESOLVED"
    c_caution = analyse_community(com, build_debates(rows_c, votes, com), draws=50, seed=13)
    check("S2-codable share 16/20 alone gives READ_WITH_CAUTION with graphs still written",
          c_caution["status"] == "READ_WITH_CAUTION" and "populations" in c_caution
          and c_caution["populations"]["s2_codable"]["n"] == 16)

    # a Dilemmas-shaped community: every codable verdict is one-loser, no both-party stratum
    dcom = COMMUNITY_BY_KEY["grok_dilemmas"]
    drows, dvotes = [], []
    for i in range(8):
        v = "ACTION_A" if i % 2 == 0 else "ACTION_B"
        obj = (1, 0, 0) if v == "ACTION_A" else (0, 1, 0)
        drows.append({"model": GROK, "item_id": f"d{i}", "arm": "neutral", "sample_idx": "0",
                      "gold_verdict": "ACTION_A", "verdict": v, "synthesis_verdict": v,
                      "n_objectors": "1", "n_r3_unparsed": "0"})
        for s, o in zip(dcom.seats, obj):
            dvotes.append({"model": GROK, "item_id": f"d{i}", "arm": "neutral", "sample_idx": "0",
                           "role_id": s, "objected_r3": str(o)})
    dc = analyse_community(dcom, build_debates(drows, dvotes, dcom), draws=50, seed=13)
    DG = dc["populations"]["s2_codable"]["graphs"]
    check("dilemmas: one_loser stratum equals the whole population, advocate phi -1, no both_party key",
          DG["one_loser"]["n"] == 8 and DG["all"]["n"] == 8 and "both_party" not in DG
          and DG["one_loser"]["edges"]["advocate_1|advocate_2"]["phi"]["point"] == -1.0
          and dc["populations"]["s2_codable"]["who_loses"]["ACTION_A"]["most_frequent_objector"]["position"]
          == "advocate_1")

    # a games-shaped community with zero objections: every phi undefined, every rate 0
    gcom = COMMUNITY_BY_KEY["grok_games_binding_neutral"]
    grows, gvotes = [], []
    for i in range(6):
        gt = "DILEMMA_BOTH_PREFER" if i < 3 else "ASSURANCE"
        grows.append({"model": GROK, "item_id": f"g{i}", "arm": "third_neutral", "sample_idx": "0",
                      "gold_verdict": "PLAN_11", "verdict": "PLAN_11", "synthesis_verdict": "PLAN_11",
                      "correct": "1", "gold_type": gt, "region": "value",
                      "n_objectors": "0", "n_r3_unparsed": "0"})
        for s in gcom.seats:
            gvotes.append({"model": GROK, "item_id": f"g{i}", "arm": "third_neutral", "sample_idx": "0",
                           "role_id": s, "objected_r3": "0"})
    gc = analyse_community(gcom, build_debates(grows, gvotes, gcom), draws=50, seed=13)
    GG = gc["populations"]["s2_codable"]["graphs"]
    check("games with no objection: every node rate 0 with interval [0, 0], every phi None, sensor lift None",
          all(v["objection_rate"]["point"] == 0.0 and v["objection_rate"]["lo"] == 0.0
              for v in GG["all"]["nodes"].values())
          and all(v["phi"]["point"] is None and v["phi"]["lo"] is None for v in GG["all"]["edges"].values())
          and GG["dilemma_both_prefer"]["n"] == 3 and GG["assurance"]["n"] == 3
          and gc["populations"]["s1_s2_codable"]["sensor"]["strata"]["all"]["lift"]["point"] is None)

    res = analyse({"grok_aita": debates}, draws=draws, seed=13)
    check("protocol block: five nodes, seventeen edges, the flag a three-seat hyperedge, six read edges removed in the noedge cell, "
          "annotations read from the grok community (fire 0.2, one-loser ratio undefined here)",
          len(res["protocol"]["nodes"]) == 5 and len(res["protocol"]["edges"]) == 17
          and sum(1 for e in res["protocol"]["edges"] if e["kind"] == "read") == 6
          and [e for e in res["protocol"]["edges"] if e["kind"] == "flag"][0]["seats"] == ["W", "C", "N"]
          and abs(res["protocol"]["annotations"]["fire_rate_raw"] - 0.2) < 1e-12
          and res["protocol"]["annotations"]["one_loser_ratio"] is None
          and res["protocol"]["annotations"]["n_sensor_population"] == 18
          and res["protocol"]["annotations"]["routing_gain_over_solo_16_11a"]["kind"].startswith("record quote"))
    check("communities not loaded are recorded ABSENT and the guards summary names every community",
          res["communities"]["nano_aita"]["status"] == "ABSENT"
          and set(res["guards_summary"]) == {c.key for c in COMMUNITIES})
    check("no em dash in this module's docstring", "\u2014" not in (__doc__ or ""))

    print(f"\nselftest: {len(fails)} failure(s)" if fails else "\nselftest: ALL OK")
    return 1 if fails else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--only", nargs="*", default=None, help="community keys to run (default all present)")
    ap.add_argument("--out", type=Path, default=ANALYSIS_PATH)
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    loaded = {}
    for com in COMMUNITIES:
        if args.only and com.key not in args.only:
            continue
        deb = load_community(com)
        if deb is not None:
            loaded[com.key] = deb
    res = analyse(loaded, draws=args.draws, seed=args.seed)
    print_report(res)
    args.out.write_text(json.dumps(res, indent=1))
    print(f"\n  wrote {args.out}")
    return 3 if res["any_structural_guard_failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
