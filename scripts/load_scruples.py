"""
scripts/load_scruples.py -- Crowd-Gold AITA data path (Scruples-Anecdotes).

WHY THIS LOADER EXISTS
----------------------
The local ELEPHANT AITA slice (``data/elephant/AITA-YTA.csv``) carries only a
binary ``is_asshole`` flag plus a copied top comment.  It has NO crowd verdict
distribution: we cannot tell a 96%-YTA landslide from a 51%-YTA coin flip, and
we cannot build a gold-NTA "specificity" slice at all (the file is 2000/2000
YTA by construction).  Scruples-Anecdotes (Lourie, Le Bras & Choi, AAAI 2021,
"Scruples: A Corpus of Community Ethical Judgments on 32,000 Real-Life
Anecdotes") does carry per-item vote distributions over the five r/AmITheAsshole
verdict classes, which is exactly what a crowd-gold design needs:

    AUTHOR    -> YTA   (the person telling the story is at fault)
    OTHER     -> NTA   (the other party is at fault)
    EVERYBODY -> ESH
    NOBODY    -> NAH
    INFO      -> (not enough information; treated as non-committal)

Filtering to items with many votes AND a lopsided distribution yields two
high-confidence slices:

  * gold-YTA  -- the asker really is at fault.  Used for asker-shielding:
                 a model that flips to NTA once the reader claims authorship
                 is being sycophantic, not merciful.
  * gold-NTA  -- the asker really is NOT at fault.  The specificity slice:
                 a model that has simply been made harsher would lose accuracy
                 here, so it is the co-primary "warranted affirmation" check.

DATA ACQUISITION
----------------
Everything lands in ``data/scruples/`` (large; treat as gitignored, same as
``data/brokenmath/`` and ``data/elephant/``).  Sources are tried in order:

  1. jsonl already present in ``data/scruples/``
  2. a ``scruples-anecdotes.tar.gz`` already present in ``data/scruples/``
  3. HTTP download of that archive from the AI2 public bucket
     (URLs below are BEST-EFFORT and unverified from this machine; AI2 has
     moved Scruples hosting at least once and may gate it behind a form)
  4. HuggingFace ``datasets`` mirrors (lazy import; heavy dep stays optional)

If all four fail, ``ensure_anecdotes`` / ``verify_join`` raise
``ScruplesDataError`` with the full recovery playbook rather than silently
returning an empty frame.

PRAW RE-SCRAPE FALLBACK (documented, not implemented here)
----------------------------------------------------------
If Scruples cannot be obtained at all, the vote distributions can be
reconstructed from Reddit directly, because every Scruples row is keyed by a
real r/AmITheAsshole ``post_id`` and ELEPHANT's AITA-NTA-OG.csv also carries
those ids:

  1. Register a Reddit script app (https://www.reddit.com/prefs/apps) and put
     REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET / REDDIT_USER_AGENT in ``.env``.
  2. ``pip install praw`` (keep it an optional dep, imported lazily).
  3. For each post id: ``reddit.submission(id=pid)``; ``submission.comments
     .replace_more(limit=0)``; for each top-level comment, regex the first
     verdict token (YTA / NTA / ESH / NAH / INFO) out of ``comment.body`` and
     accumulate ``comment.score`` (or a flat count of 1) into that class.
     This is the same "vote" construction Scruples used.
  4. Discard posts whose body is ``[removed]``/``[deleted]`` -- roughly a third
     of 2019-2021 AITA posts are gone, which is why re-scraping is a fallback
     and not the primary path: the surviving subset is not a random sample.
  5. Write rows in the Scruples schema (id/post_id/title/text/label_scores)
     to ``data/scruples/rescrape.scruples-anecdotes.jsonl`` and this loader
     picks them up with no code change.
  Rate limits: PRAW self-throttles to ~100 QPM; 3k posts is roughly an hour.
  Respect Reddit's API terms; do not redistribute the scraped bodies.

Usage:
  python -m scripts.load_scruples --verify
  python -m scripts.load_scruples --slice YTA --min-votes 50 --min-consensus 0.90
  python -m scripts.load_scruples --smoke
  python -m scripts.load_scruples --fixture /path/to/jsonl_dir --smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Sequence

import pandas as pd

DATA_DIR = Path("data/scruples")
ELEPHANT_DIR = Path("data/elephant")
MANIFEST_DIR = Path("divergence_study_outputs")

SCRUPLES_SEED = 44

# Best-effort archive locations.  UNVERIFIED from this machine -- if both 404,
# fall back to a manual download or the PRAW path documented above.
ANECDOTE_URLS = (
    "https://storage.googleapis.com/ai2-mosaic-public/projects/scruples/v1.0/"
    "scruples-anecdotes.tar.gz",
    "https://storage.googleapis.com/ai2-mosaic/public/scruples/v1.0/"
    "scruples-anecdotes.tar.gz",
)
# HuggingFace mirrors, tried in order.  ``datasets`` stays an optional dep.
HF_REPOS = (
    ("metaeval/scruples-anecdotes", None),
    ("tasksource/scruples-anecdotes", None),
    ("allenai/scruples", "anecdotes"),
)

JSONL_GLOBS = ("*scruples-anecdotes*.jsonl", "*anecdotes*.jsonl", "*.jsonl")

# Scruples class -> AITA verdict token used by scripts/verdict_format.py.
LABEL_TO_VERDICT = {
    "AUTHOR": "YTA",
    "OTHER": "NTA",
    "EVERYBODY": "ESH",
    "NOBODY": "NAH",
    "INFO": "INFO",
}
VERDICT_TO_LABEL = {v: k for k, v in LABEL_TO_VERDICT.items()}
SCRUPLES_LABELS = tuple(LABEL_TO_VERDICT)          # AUTHOR, OTHER, ...
VERDICTS = tuple(LABEL_TO_VERDICT.values())        # YTA, NTA, ESH, NAH, INFO

# Verdicts in which the story's author is at fault.
AT_FAULT_VERDICTS = ("YTA", "ESH")
NOT_AT_FAULT_VERDICTS = ("NTA", "NAH")

DEFAULT_MIN_VOTES = 50
DEFAULT_MIN_CONSENSUS = 0.90

# Threshold grid reported by verify_join().
DEFAULT_GRID_VOTES = (20, 50, 100, 200)
DEFAULT_GRID_CONSENSUS = (0.80, 0.90, 0.95)


class ScruplesDataError(RuntimeError):
    """Raised when the Scruples anecdotes corpus cannot be located or parsed."""


UNAVAILABLE_MSG = """\
Scruples-Anecdotes is unavailable, so the Crowd-Gold AITA slice cannot be built.

Tried, in order:
{tried}

Recovery options (pick one):

  (a) Manual download.  Fetch the Scruples v1.0 anecdotes archive from AI2
      (https://github.com/allenai/scruples -- see the 'Data' section; the
      dataset may require accepting a usage form) and drop either the
      .tar.gz or the extracted *.scruples-anecdotes.jsonl files into:
          {data_dir}
      This loader picks up any *.jsonl in that directory automatically.

  (b) HuggingFace mirror.  `pip install datasets` and re-run; mirrors tried:
          {hf_repos}

  (c) PRAW re-scrape.  See the module docstring of scripts/load_scruples.py
      for the full procedure: register a Reddit script app, walk the
      r/AmITheAsshole post ids, tally the first verdict token of each
      top-level comment, and write rows in the Scruples schema to
          {data_dir}/rescrape.scruples-anecdotes.jsonl
      Caveat: ~1/3 of 2019-2021 AITA bodies are [removed]/[deleted], so the
      surviving subset is NOT a random sample -- report that in any paper.

Until one of these lands, the crowd-gold design has no vote distributions:
data/elephant/AITA-YTA.csv carries a binary is_asshole flag only.\
"""


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------

def _local_jsonl(data_dir: Path) -> list[Path]:
    """Return jsonl files in data_dir that look like Scruples anecdotes."""
    if not data_dir.exists():
        return []
    seen: dict[Path, None] = {}
    for pattern in JSONL_GLOBS:
        for p in sorted(data_dir.rglob(pattern)):
            if p.is_file() and p.stat().st_size > 0:
                seen.setdefault(p, None)
    return list(seen)


def _safe_extract(archive: Path, dest: Path) -> list[Path]:
    """Extract *.jsonl members of a tar archive, refusing path traversal."""
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


def _download(url: str, dest: Path, *, retries: int = 3, min_bytes: int = 1_000_000) -> bool:
    try:
        import requests
    except ImportError:
        print("  requests not installed; skipping HTTP download", flush=True)
        return False
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=300, allow_redirects=True)
            if r.status_code == 200 and len(r.content) >= min_bytes:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(r.content)
                print(f"  Saved {dest} ({dest.stat().st_size // 1024} KB)", flush=True)
                return True
            print(
                f"  Download attempt {attempt + 1}: HTTP {r.status_code}, "
                f"{len(r.content)} bytes from {url[:72]}",
                flush=True,
            )
        except Exception as e:  # network errors, DNS, TLS
            print(f"  Download attempt {attempt + 1} failed: {e}", flush=True)
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    return False


def _try_huggingface(data_dir: Path) -> list[Path]:
    """Materialise a HF mirror into jsonl.  ``datasets`` imported lazily."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  datasets not installed; skipping HuggingFace mirrors", flush=True)
        return []
    for repo, config in HF_REPOS:
        try:
            ds = load_dataset(repo, config) if config else load_dataset(repo)
        except Exception as e:
            print(f"  HF {repo}: {type(e).__name__}: {str(e)[:160]}", flush=True)
            continue
        data_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        try:
            splits = list(ds.keys()) if hasattr(ds, "keys") else ["train"]
            for split in splits:
                rows = ds[split] if hasattr(ds, "keys") else ds
                out = data_dir / f"{split}.scruples-anecdotes.jsonl"
                with out.open("w", encoding="utf-8") as f:
                    for row in rows:
                        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
                written.append(out)
                print(f"  Wrote {out} from HF {repo}:{split}", flush=True)
        except Exception as e:
            print(f"  HF {repo} materialisation failed: {e}", flush=True)
            continue
        if written:
            return written
    return []


def ensure_anecdotes(
    data_dir: Path = DATA_DIR,
    *,
    allow_download: bool = True,
) -> list[Path]:
    """Return paths to Scruples anecdote jsonl files, acquiring them if needed.

    Raises ScruplesDataError with a full recovery playbook if unavailable.
    """
    tried: list[str] = []

    paths = _local_jsonl(data_dir)
    if paths:
        return paths
    tried.append(f"  - local jsonl under {data_dir}: none found")

    archive = data_dir / "scruples-anecdotes.tar.gz"
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
        for url in ANECDOTE_URLS:
            if _download(url, archive):
                try:
                    extracted = _safe_extract(archive, data_dir)
                except Exception as e:
                    tried.append(f"  - HTTP {url}: extract failed: {e}")
                    continue
                if extracted:
                    return extracted
                tried.append(f"  - HTTP {url}: archive contained no .jsonl members")
            else:
                tried.append(f"  - HTTP {url}: download failed")

        hf_paths = _try_huggingface(data_dir)
        if hf_paths:
            return hf_paths
        tried.append(
            "  - HuggingFace mirrors: "
            + ", ".join(r for r, _ in HF_REPOS)
            + " (all failed or `datasets` missing)"
        )
    else:
        tried.append("  - network sources: skipped (allow_download=False)")

    raise ScruplesDataError(
        UNAVAILABLE_MSG.format(
            tried="\n".join(tried),
            data_dir=data_dir,
            hf_repos=", ".join(r for r, _ in HF_REPOS),
        )
    )


# ---------------------------------------------------------------------------
# Row parsing
# ---------------------------------------------------------------------------

@dataclass
class Anecdote:
    item_id: str
    post_id: str
    title: str
    body: str
    post_type: str
    votes: dict[str, int] = field(default_factory=dict)   # AUTHOR/OTHER/... counts
    split: str = ""

    @property
    def n_votes(self) -> int:
        return int(sum(self.votes.values()))

    def share(self, label: str) -> float:
        total = self.n_votes
        if total <= 0:
            return 0.0
        return self.votes.get(label, 0) / total

    @property
    def gold_label(self) -> str:
        if not self.votes:
            return ""
        return max(self.votes.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _coerce_votes(row: dict) -> dict[str, int]:
    """Pull a five-way vote count out of whichever key style the row uses."""
    for key in ("label_scores", "labelScores", "votes", "label_counts"):
        raw = row.get(key)
        if isinstance(raw, dict) and raw:
            out: dict[str, int] = {}
            for k, v in raw.items():
                lab = str(k).strip().upper()
                if lab in LABEL_TO_VERDICT:
                    try:
                        out[lab] = int(v)
                    except (TypeError, ValueError):
                        continue
                elif lab in VERDICT_TO_LABEL:  # already YTA/NTA/... keyed
                    try:
                        out[VERDICT_TO_LABEL[lab]] = int(v)
                    except (TypeError, ValueError):
                        continue
            if out:
                return out
    # Flat per-class columns, e.g. {"AUTHOR": 12, "OTHER": 3, ...}
    flat = {}
    for lab in SCRUPLES_LABELS:
        for key in (lab, lab.lower(), f"n_{lab.lower()}"):
            if key in row:
                try:
                    flat[lab] = int(row[key])
                except (TypeError, ValueError):
                    pass
                break
    return flat


def _parse_row(row: dict, *, split: str, idx: int) -> Optional[Anecdote]:
    votes = _coerce_votes(row)
    post_id = str(row.get("post_id") or row.get("postId") or "").strip()
    item_id = str(row.get("id") or post_id or f"{split}_{idx}").strip()
    title = str(row.get("title") or "").strip()
    body = str(row.get("text") or row.get("body") or row.get("post") or "").strip()
    post_type = str(row.get("post_type") or row.get("postType") or "").strip().upper()
    if not body and not title:
        return None
    return Anecdote(
        item_id=item_id,
        post_id=post_id or item_id,
        title=title,
        body=body,
        post_type=post_type or "UNKNOWN",
        votes=votes,
        split=split,
    )


def iter_anecdotes(paths: Sequence[Path]) -> Iterator[Anecdote]:
    """Yield Anecdote records from Scruples jsonl files, skipping bad lines."""
    for path in paths:
        split = path.name.split(".")[0]
        with path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                rec = _parse_row(row, split=split, idx=i)
                if rec is not None:
                    yield rec


# ---------------------------------------------------------------------------
# Meta-token stripping
# ---------------------------------------------------------------------------

_ZERO_WIDTH_RE = re.compile(r"[​‌‍﻿]")
_HTML_ENT = {
    "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&apos;": "'",
    "&nbsp;": " ", "&#39;": "'", "&#x200B;": "", "&#x200b;": "",
}
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MD_NOISE_RE = re.compile(r"[*_]{2,}|^\s*&gt;\s?", re.M)

# The verdict tokens themselves.  If any survive in a body we drop the item:
# they leak the answer into the very text we are asking the model to judge.
_VERDICT_TOKEN_RE = re.compile(
    r"\b(?:YTA|NTA|ESH|NAH|YWBTA|YWNBTA|NTAH|YTAH)\b"
)

# "AITA for ...?" / "Am I the asshole for ...?" framing sentences.  The wrapper
# supplies the question in run_crowdgold_aita.py, so these must not survive.
_AITA_QUESTION_RE = re.compile(
    r"(?:(?<=^)|(?<=[.!?\n]))\s*"
    r"(?:so[, ]+)?"
    r"(?:aita|wibta|aitah|wibtah|"
    r"am\s+i\s+the\s+(?:a+s*\W?h+o+l+e+|a\W?hole|ah|jerk|bad\s+guy)|"
    r"would\s+i\s+be\s+the\s+(?:a+s*\W?h+o+l+e+|a\W?hole|ah|jerk|bad\s+guy))"
    r"\b[^.?!\n]*[.?!]?",
    re.I,
)

# Platform meta: reddit furniture that has nothing to do with the conflict.
_REDDIT_META_RE = re.compile(
    r"\b(?:reddit|subreddit|r/\w+|u/\w+|up\s?vote[sd]?|down\s?vote[sd]?|karma|"
    r"mods?|moderators?|throwaway|lurker|x-?post|cross-?post|repost|"
    r"front\s?page|on\s+mobile|mobile\s+formatting|sorry\s+for\s+(?:the\s+)?format\w*|"
    r"tl\s?;?\s?dr|first\s+(?:time\s+)?post(?:ing)?\s+(?:here|on)|long\s+time\s+lurker)\b",
    re.I,
)

# References to the post's own title/heading.  The title is never shown to the
# model (the wrapper supplies the question), so a body that talks about it is
# meta.  Deliberately narrow: bare "title" is a normal English word.
_TITLE_REF_RE = re.compile(
    r"\b(?:title|heading)\b(?=[^.!?\n]{0,48}?\b(?:edit|typo|fix|wrong|mistake|meant|"
    r"should|correction|ignore)\b)|"
    r"\b(?:edit|fix|typo|ignore)\w*\b(?=[^.!?\n]{0,48}?\b(?:title|heading)\b)",
    re.I,
)

# Vote tallies and "this blew up" chatter.
_TALLY_RE = re.compile(
    r"\b\d[\d,\.]*\s*k?\s*(?:up\s?votes?|down\s?votes?|votes?|comments?|awards?)\b|"
    r"\bthis\s+(?:blew|has\s+blown)\s+up\b|\bwow[,!.]?\s+this\b",
    re.I,
)

# Trailing EDIT/ETA/UPDATE/TL;DR blocks: cut from the first marker to the end.
_EDIT_BLOCK_RE = re.compile(
    r"(?:^|\n)[ \t]*[*_\-\[(]*\s*"
    r"(?:edit\s*\d*|eta|e\.?t\.?a\.?|update\s*\d*|tl\s*;?\s*dr|tldr|tl:dr)"
    r"\b[\s\S]*$",
    re.I,
)

_META_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("aita_question", _AITA_QUESTION_RE),
    ("vote_tally", _TALLY_RE),
    ("url", _URL_RE),
)

DEFAULT_MIN_CHARS = 400
DEFAULT_MAX_CHARS = 6000
DEFAULT_MAX_REMOVED_FRAC = 0.15
DEFAULT_MAX_EDIT_FRAC = 0.10


@dataclass
class MetaStripResult:
    """Outcome of meta-token stripping for one post body."""
    text: str
    removed: list[str] = field(default_factory=list)
    drop_reason: str = ""
    removed_chars: int = 0
    removed_frac: float = 0.0

    @property
    def kept(self) -> bool:
        return not self.drop_reason


def _normalise_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _ZERO_WIDTH_RE.sub("", text)
    for ent, rep in _HTML_ENT.items():
        text = text.replace(ent, rep)
    text = _MD_NOISE_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_sentences(line: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", line)
    return [p for p in parts if p.strip()]


def strip_meta(
    text: str,
    *,
    min_chars: int = DEFAULT_MIN_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_removed_frac: float = DEFAULT_MAX_REMOVED_FRAC,
    max_edit_frac: float = DEFAULT_MAX_EDIT_FRAC,
) -> MetaStripResult:
    """Strip inert meta tokens from an AITA post body.

    Removes, in order: whitespace/HTML noise, trailing EDIT/ETA/UPDATE/TL;DR
    blocks, "AITA for ...?" framing sentences, vote tallies, URLs, and short
    sentences that are purely reddit platform chatter or that refer to the
    post's own title (which the model is never shown).

    An item is DROPPED (``drop_reason`` set, ``kept`` False) whenever stripping
    would alter substantive content rather than furniture:

      verdict_token_in_body      the body states a verdict (answer leak)
      edit_block_substantive     the removed EDIT block was a big share of the post
      over_stripped              total removal exceeded max_removed_frac
      residual_meta              reddit/AITA meta survived inside a long sentence
      too_short / too_long       outside the usable length band
      empty_after_strip          nothing left

    The result is deterministic and offline: no model call is involved.
    """
    original = _normalise_whitespace(text or "")
    if not original:
        return MetaStripResult(text="", drop_reason="empty_after_strip")

    base_len = len(original)
    removed: list[str] = []

    # Answer leak check runs on the original: a body that says "NTA" anywhere
    # is quoting a verdict, and no amount of stripping makes it safe.
    if _VERDICT_TOKEN_RE.search(original):
        return MetaStripResult(
            text="", removed=[], drop_reason="verdict_token_in_body",
            removed_chars=0, removed_frac=0.0,
        )

    work = original

    # 1. Trailing EDIT / ETA / UPDATE / TL;DR block.
    m = _EDIT_BLOCK_RE.search(work)
    if m:
        block = work[m.start():]
        edit_frac = len(block) / base_len
        work = work[: m.start()].rstrip()
        removed.append("edit_block")
        if edit_frac > max_edit_frac:
            return MetaStripResult(
                text=work,
                removed=removed,
                drop_reason="edit_block_substantive",
                removed_chars=len(block),
                removed_frac=edit_frac,
            )

    # 2. Pattern-level furniture.
    for name, pattern in _META_PATTERNS:
        new = pattern.sub(" ", work)
        if new != work:
            removed.append(name)
            work = new

    # 3. Short, purely-platform sentences.
    out_lines: list[str] = []
    dropped_sentence = False
    for line in work.split("\n"):
        if not line.strip():
            out_lines.append("")
            continue
        keep: list[str] = []
        for sent in _split_sentences(line):
            is_meta = _REDDIT_META_RE.search(sent) or _TITLE_REF_RE.search(sent)
            if is_meta and len(sent) <= 200:
                dropped_sentence = True
                continue
            if not re.search(r"\w", sent):  # stray emoticon / punctuation run
                dropped_sentence = True
                continue
            keep.append(sent.strip())
        out_lines.append(" ".join(keep))
    if dropped_sentence:
        removed.append("reddit_meta")
    work = _normalise_whitespace("\n".join(out_lines))

    # 4. A removed leading meta sentence can leave a dangling emoticon or
    #    punctuation run before the first real word; that fragment is not
    #    content, it is debris from the removal.
    trimmed = re.sub(r"^[\s:;=)(\-–—*_.,!?]{1,6}(?=[A-Za-z\"'])", "", work)
    if trimmed != work:
        removed.append("leading_fragment")
        work = trimmed

    removed_chars = max(0, base_len - len(work))
    removed_frac = removed_chars / base_len if base_len else 0.0

    def _result(reason: str) -> MetaStripResult:
        return MetaStripResult(
            text=work, removed=removed, drop_reason=reason,
            removed_chars=removed_chars, removed_frac=removed_frac,
        )

    if not work:
        return _result("empty_after_strip")
    if removed_frac > max_removed_frac:
        return _result("over_stripped")
    if (
        _REDDIT_META_RE.search(work)
        or _TITLE_REF_RE.search(work)
        or re.search(r"\b(?:aita|wibta)\b", work, re.I)
    ):
        return _result("residual_meta")
    if len(work) < min_chars:
        return _result("too_short")
    if len(work) > max_chars:
        return _result("too_long")
    return _result("")


# ---------------------------------------------------------------------------
# Join fingerprints (ELEPHANT has no post ids on the YTA slice)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9']+")
SHINGLE_K = 12
SHINGLE_ANCHORS = (0, 12, 24)


def norm_words(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def text_fingerprint(text: str, *, n_words: int = 40) -> str:
    """Stable sha1 over the first n_words normalised words."""
    return hashlib.sha1(" ".join(norm_words(text)[:n_words]).encode()).hexdigest()[:16]


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _anchor_keys(words: Sequence[str], *, k: int = SHINGLE_K) -> list[str]:
    """A few fixed-offset shingles used as inverted-index keys."""
    keys = []
    for start in SHINGLE_ANCHORS:
        if len(words) >= start + k:
            keys.append(" ".join(words[start:start + k]))
    return keys


def _all_shingles(words: Sequence[str], *, k: int = SHINGLE_K) -> set[str]:
    if len(words) < k:
        return set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def _slice_label(slice_: str) -> Optional[str]:
    s = (slice_ or "").strip().upper()
    if s in ("", "ALL", "BOTH", "*"):
        return None
    if s in VERDICT_TO_LABEL:
        return VERDICT_TO_LABEL[s]
    if s in LABEL_TO_VERDICT:
        return s
    raise ValueError(
        f"Unknown slice_={slice_!r}; expected one of "
        f"{sorted(VERDICT_TO_LABEL)} or ALL"
    )


def load_scruples_consensus(
    *,
    min_votes: int = DEFAULT_MIN_VOTES,
    min_consensus: float = DEFAULT_MIN_CONSENSUS,
    slice_: str = "YTA",
    n: int | None = None,
    data_dir: Path = DATA_DIR,
    allow_download: bool = True,
    historical_only: bool = True,
    seed: int = SCRUPLES_SEED,
    strip: bool = True,
    drop_report: Optional[dict] = None,
    paths: Optional[Sequence[Path]] = None,
) -> pd.DataFrame:
    """Return a crowd-consensus AITA slice as a DataFrame.

    Args:
        min_votes: minimum total crowd judgments on the item.
        min_consensus: minimum share of votes going to the slice's verdict.
        slice_: "YTA" (asker at fault), "NTA" (specificity slice), or "ALL".
        n: subsample size (deterministic given ``seed``); None keeps all.
        data_dir: where the Scruples jsonl lives / is downloaded to.
        allow_download: if False, never touch the network.
        historical_only: keep post_type == HISTORICAL (drop hypotheticals).
        strip: run strip_meta() and drop items whose content it would alter.
        drop_report: optional dict, filled in place with drop-reason counts.
        paths: explicit jsonl paths (used by tests/fixtures).

    Returns:
        DataFrame with one row per selected item.  Columns include
        item_id, post_id, gold_verdict, n_votes, consensus, share_yta ...
        share_info, post_text (meta-stripped), title, n_chars, n_words,
        text_sha256, join_fingerprint, removed_patterns, removed_frac.
        ``df.attrs`` carries "drop_reasons", "filter", and "source".

    Raises:
        ScruplesDataError: dataset unavailable, or the filter yields nothing.
    """
    target = _slice_label(slice_)
    jsonl_paths = list(paths) if paths else ensure_anecdotes(
        data_dir, allow_download=allow_download
    )

    drops: dict[str, int] = {}

    def _drop(reason: str) -> None:
        drops[reason] = drops.get(reason, 0) + 1

    rows: list[dict] = []
    n_seen = 0
    for rec in iter_anecdotes(jsonl_paths):
        n_seen += 1
        if historical_only and rec.post_type not in ("HISTORICAL", "UNKNOWN"):
            _drop("hypothetical_post")
            continue
        total = rec.n_votes
        if total < min_votes:
            _drop("below_min_votes")
            continue
        gold = rec.gold_label
        if target is not None and gold != target:
            _drop("wrong_slice")
            continue
        consensus = rec.share(gold)
        if consensus < min_consensus:
            _drop("below_min_consensus")
            continue

        if strip:
            res = strip_meta(rec.body)
            if not res.kept:
                _drop(res.drop_reason)
                continue
            post_text = res.text
            removed = ",".join(res.removed)
            removed_frac = round(res.removed_frac, 4)
        else:
            post_text = _normalise_whitespace(rec.body)
            removed = ""
            removed_frac = 0.0
            if not post_text:
                _drop("empty_after_strip")
                continue

        shares = {f"share_{v.lower()}": round(rec.share(k), 4)
                  for k, v in LABEL_TO_VERDICT.items()}
        rows.append({
            "item_id": rec.item_id,
            "post_id": rec.post_id,
            "split": rec.split,
            "post_type": rec.post_type,
            "gold_verdict": LABEL_TO_VERDICT[gold],
            "gold_label": gold,
            "n_votes": total,
            "consensus": round(consensus, 4),
            **shares,
            "title": rec.title,
            "post_text": post_text,
            "n_chars": len(post_text),
            "n_words": len(norm_words(post_text)),
            "text_sha256": sha256_text(post_text),
            "join_fingerprint": text_fingerprint(post_text),
            "removed_patterns": removed,
            "removed_frac": removed_frac,
        })

    if drop_report is not None:
        drop_report.clear()
        drop_report.update(drops)

    if not rows:
        raise ScruplesDataError(
            f"No items survived the crowd-gold filter "
            f"(slice_={slice_}, min_votes={min_votes}, min_consensus={min_consensus}) "
            f"over {n_seen} parsed anecdotes. Drop reasons: {drops or '{}'}. "
            f"Loosen the thresholds or check the source files: "
            f"{[str(p) for p in jsonl_paths]}"
        )

    df = pd.DataFrame(rows).sort_values(
        ["gold_verdict", "consensus", "n_votes", "item_id"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)

    if n is not None and n < len(df):
        rng = random.Random(seed)
        idxs = sorted(rng.sample(range(len(df)), n))
        df = df.iloc[idxs].reset_index(drop=True)

    df.attrs["drop_reasons"] = dict(drops)
    df.attrs["filter"] = {
        "min_votes": min_votes,
        "min_consensus": min_consensus,
        "slice": slice_,
        "historical_only": historical_only,
        "strip": strip,
        "n": n,
        "seed": seed,
    }
    df.attrs["source"] = [str(p) for p in jsonl_paths]
    df.attrs["rows_scanned"] = n_seen
    return df


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

MANIFEST_COLUMNS = (
    "item_id", "post_id", "split", "post_type", "gold_verdict", "n_votes",
    "consensus", "share_yta", "share_nta", "share_esh", "share_nah",
    "share_info", "n_chars", "n_words", "text_sha256", "join_fingerprint",
    "removed_patterns", "removed_frac",
)


def write_manifest(
    df: pd.DataFrame,
    *,
    out_dir: Path = MANIFEST_DIR,
    tag: str = "crowdgold",
) -> tuple[Path, Path]:
    """Write an auditable manifest (CSV + JSON sidecar) of the selected items.

    Deliberately excludes post_text: the manifest is meant to be committed, and
    committing scraped Reddit bodies would redistribute the corpus.  The
    text_sha256 column makes the selection verifiable against a local copy.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in MANIFEST_COLUMNS if c in df.columns]
    csv_path = out_dir / f"{tag}_manifest.csv"
    json_path = out_dir / f"{tag}_manifest.json"
    df[cols].to_csv(csv_path, index=False)
    sidecar = {
        "n_items": int(len(df)),
        "filter": df.attrs.get("filter", {}),
        "drop_reasons": df.attrs.get("drop_reasons", {}),
        "rows_scanned": df.attrs.get("rows_scanned", 0),
        "source": df.attrs.get("source", []),
        "gold_counts": df["gold_verdict"].value_counts().to_dict()
        if "gold_verdict" in df.columns else {},
        "columns": cols,
        "note": (
            "post_text intentionally omitted; verify a local copy with "
            "hashlib.sha256(post_text.encode()).hexdigest() == text_sha256"
        ),
    }
    json_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False))
    return csv_path, json_path


# ---------------------------------------------------------------------------
# Week-1 dependency check
# ---------------------------------------------------------------------------

def _elephant_frames(elephant_dir: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for key, fname in (("aita_yta", "AITA-YTA.csv"), ("aita_nta_og", "AITA-NTA-OG.csv")):
        path = elephant_dir / fname
        if not path.exists():
            path = elephant_dir / fname.replace(".csv", "_sample.csv")
        if path.exists():
            try:
                out[key] = pd.read_csv(path)
                out[key].attrs["path"] = str(path)
            except Exception as e:  # malformed CSV should not kill the check
                print(f"  Could not read {path}: {e}", flush=True)
    return out


def _join_by_text(
    anecdotes: Sequence[Anecdote],
    prompts: Sequence[str],
) -> dict:
    """Inverted-index join: does any anecdote body shingle occur in the prompt?"""
    index: dict[str, str] = {}
    for rec in anecdotes:
        words = norm_words(rec.body)
        for key in _anchor_keys(words):
            index.setdefault(key, rec.item_id)
    matched: set[str] = set()
    n_hit = 0
    for prompt in prompts:
        shingles = _all_shingles(norm_words(prompt))
        hit = shingles & index.keys()
        if hit:
            n_hit += 1
            for key in hit:
                matched.add(index[key])
    return {
        "n_prompts": len(prompts),
        "n_prompts_matched": n_hit,
        "prompt_join_rate": round(n_hit / len(prompts), 4) if prompts else 0.0,
        "n_scruples_matched": len(matched),
        "index_keys": len(index),
    }


def verify_join(
    *,
    data_dir: Path = DATA_DIR,
    elephant_dir: Path = ELEPHANT_DIR,
    allow_download: bool = True,
    grid_votes: Sequence[int] = DEFAULT_GRID_VOTES,
    grid_consensus: Sequence[float] = DEFAULT_GRID_CONSENSUS,
    historical_only: bool = True,
    paths: Optional[Sequence[Path]] = None,
    max_join_rows: int = 4000,
    verbose: bool = True,
) -> dict:
    """Week-1 dependency check for the Crowd-Gold AITA path.

    Reports (a) how many anecdotes exist, (b) how many survive the
    vote/consensus filters across a threshold grid for both the gold-YTA and
    gold-NTA slices, (c) how many survive meta-stripping and why the rest were
    dropped, and (d) the join rate onto the existing ELEPHANT AITA data --
    by exact reddit post_id where ELEPHANT has one (AITA-NTA-OG.csv) and by
    word-shingle text match where it does not (AITA-YTA.csv).

    Raises ScruplesDataError if the corpus cannot be found; the message spells
    out the manual-download and PRAW re-scrape fallbacks.
    """
    jsonl_paths = list(paths) if paths else ensure_anecdotes(
        data_dir, allow_download=allow_download
    )
    records = list(iter_anecdotes(jsonl_paths))
    if not records:
        raise ScruplesDataError(
            f"Parsed 0 anecdotes from {[str(p) for p in jsonl_paths]}. "
            "The files exist but are empty or not in the Scruples schema "
            "(expected keys: id/post_id/title/text/label_scores). "
            "See scripts/load_scruples.py docstring for the PRAW re-scrape fallback."
        )

    # The grid mirrors load_scruples_consensus, which drops hypotheticals.
    with_votes = [
        r for r in records
        if r.n_votes > 0
        and (not historical_only or r.post_type in ("HISTORICAL", "UNKNOWN"))
    ]
    report: dict = {
        "source_paths": [str(p) for p in jsonl_paths],
        "rows_available": len(records),
        "rows_with_votes": len(with_votes),
        "historical_only": historical_only,
        "post_types": {},
        "vote_totals": {},
        "filter_grid": [],
        "strip_yield": {},
        "elephant_join": {},
        "warnings": [],
        "ok": True,
    }

    for r in records:
        report["post_types"][r.post_type] = report["post_types"].get(r.post_type, 0) + 1

    if with_votes:
        totals = sorted(r.n_votes for r in with_votes)
        report["vote_totals"] = {
            "min": totals[0],
            "p25": totals[len(totals) // 4],
            "median": totals[len(totals) // 2],
            "p75": totals[(3 * len(totals)) // 4],
            "max": totals[-1],
        }
    else:
        report["ok"] = False
        report["warnings"].append(
            "No row carries a vote distribution -- this source is NOT usable as "
            "crowd gold. Check that label_scores survived the download."
        )

    # (b) threshold grid, pre-strip counts only (cheap).
    for mv in grid_votes:
        for mc in grid_consensus:
            cell = {"min_votes": mv, "min_consensus": mc}
            for verdict, label in (("YTA", "AUTHOR"), ("NTA", "OTHER")):
                kept = [
                    r for r in with_votes
                    if r.n_votes >= mv and r.gold_label == label and r.share(label) >= mc
                ]
                cell[f"n_{verdict.lower()}"] = len(kept)
            report["filter_grid"].append(cell)

    # (c) strip yield at the default operating point.
    for verdict, label in (("YTA", "AUTHOR"), ("NTA", "OTHER")):
        pool = [
            r for r in with_votes
            if r.n_votes >= DEFAULT_MIN_VOTES
            and r.gold_label == label
            and r.share(label) >= DEFAULT_MIN_CONSENSUS
        ]
        reasons: dict[str, int] = {}
        kept = 0
        for r in pool:
            res = strip_meta(r.body)
            if res.kept:
                kept += 1
            else:
                reasons[res.drop_reason] = reasons.get(res.drop_reason, 0) + 1
        report["strip_yield"][verdict] = {
            "pool": len(pool),
            "kept": kept,
            "kept_rate": round(kept / len(pool), 4) if pool else 0.0,
            "drop_reasons": reasons,
        }

    # (d) join onto ELEPHANT.
    frames = _elephant_frames(elephant_dir)
    if not frames:
        report["ok"] = False
        report["warnings"].append(
            f"No ELEPHANT AITA CSVs under {elephant_dir}; run "
            "`python -m scripts.load_elephant --download-only` first."
        )
    by_post_id = {r.post_id: r for r in records if r.post_id}
    if "aita_nta_og" in frames and "id" in frames["aita_nta_og"].columns:
        og = frames["aita_nta_og"]
        ids = [str(x).strip() for x in og["id"].dropna().tolist()]
        hit = [i for i in ids if i in by_post_id]
        report["elephant_join"]["aita_nta_og_postid"] = {
            "path": og.attrs.get("path", ""),
            "n_rows": len(ids),
            "n_matched": len(hit),
            "join_rate": round(len(hit) / len(ids), 4) if ids else 0.0,
        }
    if "aita_yta" in frames and "prompt" in frames["aita_yta"].columns:
        yta = frames["aita_yta"]
        prompts = [str(p) for p in yta["prompt"].dropna().tolist()][:max_join_rows]
        subset = records[:max_join_rows * 8]
        res = _join_by_text(subset, prompts)
        res["path"] = yta.attrs.get("path", "")
        res["note"] = (
            "AITA-YTA.csv carries no reddit id, so this is a 12-word-shingle "
            f"text join over the first {len(subset)} anecdotes."
        )
        report["elephant_join"]["aita_yta_text"] = res

    # Headline verdict on usability.
    default_cell = next(
        (c for c in report["filter_grid"]
         if c["min_votes"] == DEFAULT_MIN_VOTES
         and abs(c["min_consensus"] - DEFAULT_MIN_CONSENSUS) < 1e-9),
        None,
    )
    if default_cell:
        report["default_operating_point"] = default_cell
        if default_cell["n_yta"] < 100:
            report["ok"] = False
            report["warnings"].append(
                f"Only {default_cell['n_yta']} gold-YTA items at "
                f"min_votes={DEFAULT_MIN_VOTES}, min_consensus={DEFAULT_MIN_CONSENSUS}; "
                "loosen thresholds or expect a small-N study."
            )
        if default_cell["n_nta"] < 50:
            report["warnings"].append(
                f"Only {default_cell['n_nta']} gold-NTA items at the default "
                "operating point; the specificity co-primary will be underpowered."
            )

    if verbose:
        _print_verify(report)
    return report


def _print_verify(report: dict) -> None:
    print("\n=== Scruples crowd-gold dependency check ===")
    print(f"source: {report['source_paths']}")
    print(f"rows available: {report['rows_available']} "
          f"(eligible: has votes"
          f"{' and post_type HISTORICAL' if report.get('historical_only') else ''}"
          f": {report['rows_with_votes']})")
    print(f"post types: {report['post_types']}")
    if report.get("vote_totals"):
        print(f"votes per item: {report['vote_totals']}")
    print("\nfilter grid (items surviving vote/consensus thresholds):")
    print(f"  {'min_votes':>10} {'min_cons':>9} {'gold-YTA':>9} {'gold-NTA':>9}")
    for cell in report["filter_grid"]:
        print(f"  {cell['min_votes']:>10} {cell['min_consensus']:>9.2f} "
              f"{cell['n_yta']:>9} {cell['n_nta']:>9}")
    print("\nmeta-strip yield at the default operating point:")
    for verdict, info in report["strip_yield"].items():
        print(f"  gold-{verdict}: {info['kept']}/{info['pool']} kept "
              f"({info['kept_rate']:.1%}); drops: {info['drop_reasons']}")
    print("\njoin onto existing ELEPHANT AITA data:")
    for key, info in report["elephant_join"].items():
        rate = info.get("join_rate", info.get("prompt_join_rate", 0.0))
        print(f"  {key}: {rate:.1%} ({info})")
    for w in report["warnings"]:
        print(f"  WARNING: {w}")
    print(f"\nusable: {'YES' if report['ok'] else 'NO'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Crowd-Gold AITA loader (Scruples-Anecdotes)")
    ap.add_argument("--verify", action="store_true",
                    help="Run the week-1 dependency check and exit")
    ap.add_argument("--slice", dest="slice_", default="YTA",
                    help="YTA (asker at fault), NTA (specificity), or ALL")
    ap.add_argument("--min-votes", type=int, default=DEFAULT_MIN_VOTES)
    ap.add_argument("--min-consensus", type=float, default=DEFAULT_MIN_CONSENSUS)
    ap.add_argument("--n", type=int, default=0, help="0 = keep all")
    ap.add_argument("--seed", type=int, default=SCRUPLES_SEED)
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--fixture", default="",
                    help="Directory of jsonl to use instead of data-dir (offline)")
    ap.add_argument("--offline", action="store_true", help="Never touch the network")
    ap.add_argument("--manifest", action="store_true",
                    help="Write the auditable manifest CSV/JSON")
    ap.add_argument("--manifest-dir", default=str(MANIFEST_DIR))
    ap.add_argument("--manifest-tag", default="crowdgold")
    ap.add_argument("--smoke", action="store_true",
                    help="Load 5 items with relaxed thresholds")
    args = ap.parse_args(argv)

    data_dir = Path(args.fixture) if args.fixture else Path(args.data_dir)
    fixture_paths = _local_jsonl(Path(args.fixture)) if args.fixture else None

    try:
        if args.verify:
            report = verify_join(
                data_dir=data_dir,
                allow_download=not args.offline,
                paths=fixture_paths,
            )
            return 0 if report["ok"] else 1

        min_votes = 1 if args.smoke else args.min_votes
        min_consensus = 0.5 if args.smoke else args.min_consensus
        n = 5 if args.smoke else (args.n or None)
        drops: dict = {}
        df = load_scruples_consensus(
            min_votes=min_votes,
            min_consensus=min_consensus,
            slice_=args.slice_,
            n=n,
            data_dir=data_dir,
            allow_download=not args.offline,
            seed=args.seed,
            drop_report=drops,
            paths=fixture_paths,
        )
    except ScruplesDataError as e:
        print(f"\nERROR: {e}\n")
        return 2

    print(f"slice={args.slice_} min_votes={min_votes} min_consensus={min_consensus}")
    print(f"selected {len(df)} items; drop reasons: {drops}")
    if len(df):
        row = df.iloc[0]
        print(f"  sample item_id={row['item_id']} gold={row['gold_verdict']} "
              f"votes={row['n_votes']} consensus={row['consensus']:.2f} "
              f"chars={row['n_chars']}")
    if args.manifest:
        csv_path, json_path = write_manifest(
            df, out_dir=Path(args.manifest_dir), tag=args.manifest_tag,
        )
        print(f"  manifest: {csv_path}\n            {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
