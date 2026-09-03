"""
scripts/load_scruples_dilemmas.py -- Addendum 12: a second task family for
the actuator ladder, built on Scruples DILEMMAS (paired real anecdotes,
comparative "which action was more wrong" verdict) instead of Scruples
ANECDOTES (single-post at-fault verdict, the AITA family every prior
deliberation addendum in this programme has used).

WHY DILEMMAS, AND WHY IT IS NOT A SEPARATE, THINNER DATASET
-------------------------------------------------------------
Scruples Dilemmas ships as one-line action descriptions (median 47 chars
in the dev split, e.g. "asking my boyfriend to stop calling me \"hot\""),
far too terse to support the rich, multi-round embodiment protocol every
prior deliberation addendum depends on: there is no narrative for a
model to defend. But every dilemma action carries the SAME `id` as an
anecdote in scruples-anecdotes.jsonl, because Dilemmas is literally built
by pairing up two real Anecdotes and asking a FRESH set of Mechanical
Turk crowd workers "which of these two people's actions was more in the
wrong" -- same corpus, same annotation process, same trust/vetting,
genuinely different task STRUCTURE (comparative, two independent
first-person accounts) and genuinely different ground-truth definition
(relative wrongness between two strangers, not an at-fault verdict
against one narrator's own account).

Verified 2026-09-03, zero API spend: 4,678 of the dev split's 4,680
dilemma action ids (99.9%) are present in the local anecdotes corpus; the
two exceptions are both the same missing id, a real small gap rather than
a systematic mismatch. This loader recovers the full anecdote text for
both sides of each pair,
so a deliberation runner for this instrument has as much narrative
material to embody as the AITA runner does.

DATA ACQUISITION
-----------------
`ensure_dilemmas` mirrors `load_scruples.ensure_anecdotes`'s recovery
playbook: local jsonl -> local archive -> HTTP download from the AI2
public bucket (URL confirmed reachable 2026-09-03) -> ScruplesDataError
with instructions. The Anecdotes corpus must ALSO be present locally
(`load_scruples.ensure_anecdotes`) for the id-join; if it is not, this
raises loudly rather than silently falling back to the terse one-line
description.

Usage:
  python -m scripts.load_scruples_dilemmas --verify
  python -m scripts.load_scruples_dilemmas --selftest
"""
from __future__ import annotations

import argparse
import json
import random
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence

import pandas as pd

from scripts.load_scruples import (
    DATA_DIR,
    ScruplesDataError,
    _download,
    _normalise_whitespace,
    ensure_anecdotes,
    iter_anecdotes,
    norm_words,
    sha256_text,
    strip_meta,
    text_fingerprint,
)

DILEMMAS_DATA_DIR = DATA_DIR / "dilemmas"
DILEMMAS_URL = (
    "https://storage.googleapis.com/ai2-mosaic-public/projects/scruples/"
    "v1.0/data/dilemmas.tar.gz"
)
JSONL_GLOBS = ("*.scruples-dilemmas.jsonl",)
DILEMMAS_SEED = 44

UNAVAILABLE_MSG = """
Scruples DILEMMAS corpus not found and could not be downloaded.

Tried:
{tried}

Recovery:
  1. Manual download: {url}
     Extract into {data_dir}/ (four jsonl files: train, train-extra, dev, test).
  2. Or run this loader once with network access; it caches to {data_dir}/.

This is a DIFFERENT archive from scruples-anecdotes.tar.gz (already used by
scripts/load_scruples.py); both are needed -- Dilemmas for the pairing and
gold comparative label, Anecdotes for the full narrative text each side of
the pair joins back to.
"""


def _safe_extract(archive: Path, dest: Path) -> list[Path]:
    out: list[Path] = []
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tf:
        for member in tf.getmembers():
            if not member.isfile() or not member.name.endswith(".jsonl"):
                continue
            name = Path(member.name).name  # flatten; refuses ../ by construction
            if not name or name.startswith("."):
                continue
            target = dest / name
            src = tf.extractfile(member)
            if src is None:
                continue
            target.write_bytes(src.read())
            out.append(target)
    return out


def _local_jsonl(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        return []
    seen: dict[Path, None] = {}
    for pattern in JSONL_GLOBS:
        for p in sorted(data_dir.rglob(pattern)):
            if p.is_file() and p.stat().st_size > 0:
                seen.setdefault(p, None)
    return list(seen)


def ensure_dilemmas(data_dir: Path = DILEMMAS_DATA_DIR, *,
                    allow_download: bool = True) -> list[Path]:
    """Return paths to Scruples dilemmas jsonl files, acquiring them if needed."""
    tried: list[str] = []

    paths = _local_jsonl(data_dir)
    if paths:
        return paths
    tried.append(f"  - local jsonl under {data_dir}: none found")

    archive = data_dir.parent / "dilemmas.tar.gz"
    if archive.exists():
        try:
            extracted = _safe_extract(archive, data_dir)
            if extracted:
                return extracted
            tried.append(f"  - local archive {archive}: contained no .jsonl members")
        except Exception as e:
            tried.append(f"  - local archive {archive}: {type(e).__name__}: {e}")
    else:
        tried.append(f"  - local archive {archive}: not present")

    if allow_download:
        if _download(DILEMMAS_URL, archive):
            try:
                extracted = _safe_extract(archive, data_dir)
            except Exception as e:
                extracted = []
                tried.append(f"  - HTTP {DILEMMAS_URL}: extract failed: {e}")
            if extracted:
                return extracted
            tried.append(f"  - HTTP {DILEMMAS_URL}: archive contained no .jsonl members")
        else:
            tried.append(f"  - HTTP {DILEMMAS_URL}: download failed")
    else:
        tried.append("  - network sources: skipped (allow_download=False)")

    raise ScruplesDataError(
        UNAVAILABLE_MSG.format(
            tried="\n".join(tried), url=DILEMMAS_URL, data_dir=data_dir,
        )
    )


@dataclass
class DilemmaAction:
    action_id: str
    description: str        # the dilemma file's own one-line gerund phrase
    title: str               # joined from the full anecdote, "" if unjoined
    body: str                # joined from the full anecdote, "" if unjoined
    post_type: str = ""

    @property
    def joined(self) -> bool:
        return bool(self.body)


@dataclass
class DilemmaPair:
    pair_id: str
    actions: tuple  # (DilemmaAction, DilemmaAction)
    gold_label: int          # index (0/1) of the action judged MORE wrong
    gold_annotations: tuple  # raw annotator counts, same order as actions
    controversial: bool
    split: str = ""

    @property
    def n_annotators(self) -> int:
        return int(sum(self.gold_annotations))

    @property
    def agreement(self) -> float:
        n = self.n_annotators
        return (self.gold_annotations[self.gold_label] / n) if n else 0.0

    @property
    def fully_joined(self) -> bool:
        return all(a.joined for a in self.actions)


def _anecdote_index(paths: Sequence[Path]) -> dict[str, "object"]:
    """item_id -> Anecdote, built once from the local anecdotes corpus."""
    return {rec.item_id: rec for rec in iter_anecdotes(paths)}


def iter_dilemmas(
    dilemma_paths: Sequence[Path],
    anecdote_index: dict,
    *,
    strip: bool = True,
) -> Iterator[DilemmaPair]:
    for path in dilemma_paths:
        split = path.stem.split(".")[0]
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                raw_actions = row.get("actions") or []
                if len(raw_actions) != 2:
                    continue
                gold_label = row.get("gold_label")
                gold_annotations = row.get("gold_annotations")
                if gold_label not in (0, 1) or not gold_annotations:
                    continue
                built: list[DilemmaAction] = []
                for a in raw_actions:
                    aid = str(a.get("id") or "")
                    desc = str(a.get("description") or "").strip()
                    anec = anecdote_index.get(aid)
                    title, body, ptype = "", "", ""
                    if anec is not None:
                        title = anec.title
                        ptype = anec.post_type
                        if strip:
                            res = strip_meta(anec.body)
                            body = res.text if res.kept else ""
                        else:
                            body = _normalise_whitespace(anec.body)
                    built.append(DilemmaAction(
                        action_id=aid, description=desc, title=title,
                        body=body, post_type=ptype,
                    ))
                yield DilemmaPair(
                    pair_id=str(row.get("id") or ""),
                    actions=tuple(built),
                    gold_label=int(gold_label),
                    gold_annotations=tuple(int(x) for x in gold_annotations),
                    controversial=bool(row.get("controversial", False)),
                    split=split,
                )


def load_dilemma_consensus(
    *,
    n: Optional[int] = None,
    min_annotators: int = 4,
    require_full_join: bool = True,
    historical_only: bool = True,
    exclude_controversial: bool = False,
    splits: Sequence[str] = ("dev", "test", "train"),
    data_dir: Path = DILEMMAS_DATA_DIR,
    anecdotes_data_dir: Optional[Path] = None,
    allow_download: bool = True,
    seed: int = DILEMMAS_SEED,
    strip: bool = True,
    drop_report: Optional[dict] = None,
    dilemma_paths: Optional[Sequence[Path]] = None,
    anecdote_paths: Optional[Sequence[Path]] = None,
) -> pd.DataFrame:
    """Return a joined, filtered Dilemmas slice as a DataFrame, one row per
    pair, with full narrative text recovered for both sides.

    Columns: pair_id, gold_label, gold_index_a_wrong (bool), agreement,
    n_annotators, controversial, split, a_item_id, a_title, a_post_text,
    a_n_chars, b_item_id, b_title, b_post_text, b_n_chars.

    Raises ScruplesDataError if either corpus is unavailable or the filter
    yields nothing.
    """
    d_paths = list(dilemma_paths) if dilemma_paths else ensure_dilemmas(
        data_dir, allow_download=allow_download)
    a_paths = list(anecdote_paths) if anecdote_paths else ensure_anecdotes(
        anecdotes_data_dir or DATA_DIR, allow_download=allow_download)
    anec_index = _anecdote_index(a_paths)

    keep_paths = [p for p in d_paths if any(s in p.name for s in splits)]
    if not keep_paths:
        keep_paths = d_paths

    drops: dict[str, int] = {}

    def _drop(reason: str) -> None:
        drops[reason] = drops.get(reason, 0) + 1

    rows: list[dict] = []
    n_seen = 0
    for pair in iter_dilemmas(keep_paths, anec_index, strip=strip):
        n_seen += 1
        if pair.n_annotators < min_annotators:
            _drop("below_min_annotators")
            continue
        if exclude_controversial and pair.controversial:
            _drop("controversial")
            continue
        if require_full_join and not pair.fully_joined:
            _drop("action_not_joined_or_stripped_empty")
            continue
        a, b = pair.actions
        if historical_only and not (
            a.post_type in ("HISTORICAL", "UNKNOWN", "")
            and b.post_type in ("HISTORICAL", "UNKNOWN", "")
        ):
            _drop("hypothetical_post")
            continue
        rows.append({
            "pair_id": pair.pair_id,
            "split": pair.split,
            "gold_label": pair.gold_label,
            "gold_index_a_wrong": pair.gold_label == 0,
            "agreement": round(pair.agreement, 4),
            "n_annotators": pair.n_annotators,
            "controversial": pair.controversial,
            "a_item_id": a.action_id,
            "a_title": a.title,
            "a_post_text": a.body,
            "a_n_chars": len(a.body),
            "a_n_words": len(norm_words(a.body)),
            "a_text_sha256": sha256_text(a.body),
            "a_join_fingerprint": text_fingerprint(a.body) if a.body else "",
            "b_item_id": b.action_id,
            "b_title": b.title,
            "b_post_text": b.body,
            "b_n_chars": len(b.body),
            "b_n_words": len(norm_words(b.body)),
            "b_text_sha256": sha256_text(b.body),
            "b_join_fingerprint": text_fingerprint(b.body) if b.body else "",
        })

    if drop_report is not None:
        drop_report.clear()
        drop_report.update(drops)

    if not rows:
        raise ScruplesDataError(
            f"No dilemma pairs survived the filter (min_annotators="
            f"{min_annotators}, require_full_join={require_full_join}, "
            f"exclude_controversial={exclude_controversial}) over {n_seen} "
            f"parsed pairs. Drop reasons: {drops or '{}'}."
        )

    df = pd.DataFrame(rows)
    if n is not None and n < len(df):
        rng = random.Random(seed)
        idx = sorted(rng.sample(range(len(df)), n))
        df = df.iloc[idx].reset_index(drop=True)
    return df


def _selftest() -> int:
    fails: list[str] = []

    def check(name, cond):
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    try:
        d_paths = ensure_dilemmas(allow_download=False)
        a_paths = ensure_anecdotes(allow_download=False)
    except ScruplesDataError as e:
        check(f"both corpora present locally ({e})", False)
        print(f"\n{len(fails)} FAILED")
        return 1

    check("dilemmas jsonl files found", len(d_paths) >= 1)
    check("anecdotes jsonl files found", len(a_paths) >= 1)

    anec_index = _anecdote_index(a_paths)
    check("anecdote index nonempty", len(anec_index) > 0)

    dev_paths = [p for p in d_paths if "dev" in p.name]
    check("dev split present", len(dev_paths) == 1)
    pairs = list(iter_dilemmas(dev_paths, anec_index, strip=False))
    check("dev split has pairs", len(pairs) > 0)

    n_joined = sum(1 for p in pairs if p.fully_joined)
    check(f"the overwhelming majority of dev pairs' both actions join to a "
          f"full anecdote ({n_joined}/{len(pairs)} = "
          f"{n_joined / len(pairs):.4f}, a real small gap is expected, not "
          f"a systematic mismatch)", n_joined / len(pairs) >= 0.99)

    check("gold_label is always 0 or 1",
          all(p.gold_label in (0, 1) for p in pairs))
    check("gold_annotations has exactly 2 entries, matching the 2 actions",
          all(len(p.gold_annotations) == 2 for p in pairs))
    check("agreement is the winning label's own vote share",
          all(0.0 <= p.agreement <= 1.0 for p in pairs))

    gold0_share = sum(1 for p in pairs if p.gold_label == 0) / len(pairs)
    check(f"gold label is roughly balanced across the two positions "
          f"({gold0_share:.3f}, not a fixed-position artifact)",
          0.35 <= gold0_share <= 0.65)

    controversial_share = sum(1 for p in pairs if p.controversial) / len(pairs)
    check(f"controversial share is a real, nontrivial fraction "
          f"({controversial_share:.3f})", 0.2 <= controversial_share <= 0.95)

    drop_report: dict = {}
    df = load_dilemma_consensus(
        dilemma_paths=dev_paths, anecdote_paths=a_paths,
        min_annotators=4, require_full_join=True, historical_only=True,
        exclude_controversial=False, drop_report=drop_report,
    )
    check("load_dilemma_consensus returns a nonempty, fully-joined frame",
          len(df) > 0 and (df["a_n_chars"] > 0).all()
          and (df["b_n_chars"] > 0).all())
    check("returned frame's two sides are never the same anecdote",
          (df["a_item_id"] != df["b_item_id"]).all())

    df2 = load_dilemma_consensus(
        dilemma_paths=dev_paths, anecdote_paths=a_paths,
        n=20, seed=44, min_annotators=4,
    )
    df3 = load_dilemma_consensus(
        dilemma_paths=dev_paths, anecdote_paths=a_paths,
        n=20, seed=44, min_annotators=4,
    )
    check("n-subsample is deterministic given a fixed seed",
          list(df2["pair_id"]) == list(df3["pair_id"]) and len(df2) == 20)

    df_excl = load_dilemma_consensus(
        dilemma_paths=dev_paths, anecdote_paths=a_paths,
        min_annotators=4, exclude_controversial=True,
    )
    check("exclude_controversial actually removes controversial pairs",
          not df_excl["controversial"].any() and len(df_excl) < len(df))

    print(f"\n{'ALL OK' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--min-annotators", type=int, default=4)
    ap.add_argument("--exclude-controversial", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        return _selftest()

    d_paths = ensure_dilemmas()
    a_paths = ensure_anecdotes()
    anec_index = _anecdote_index(a_paths)
    for path in d_paths:
        pairs = list(iter_dilemmas([path], anec_index, strip=False))
        joined = sum(1 for p in pairs if p.fully_joined)
        gold0 = sum(1 for p in pairs if p.gold_label == 0)
        controversial = sum(1 for p in pairs if p.controversial)
        print(f"{path.name}: {len(pairs)} pairs, {joined}/{len(pairs)} fully "
              f"joined to anecdote text, gold_label==0 {gold0/len(pairs):.3f}, "
              f"controversial {controversial/len(pairs):.3f}")

    drop_report: dict = {}
    df = load_dilemma_consensus(
        min_annotators=a.min_annotators,
        exclude_controversial=a.exclude_controversial,
        drop_report=drop_report,
    )
    print(f"\nload_dilemma_consensus: {len(df)} pairs "
          f"(min_annotators={a.min_annotators}, "
          f"exclude_controversial={a.exclude_controversial})")
    print(f"drop reasons: {drop_report}")
    print(f"mean a_n_chars={df['a_n_chars'].mean():.0f} "
          f"mean b_n_chars={df['b_n_chars'].mean():.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
