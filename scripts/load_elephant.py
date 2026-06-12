"""
scripts/load_elephant.py -- ELEPHANT benchmark data loader.

Downloads and caches OSF datasets.zip into data/elephant/ (gitignored).
Uses OSF file API when the legacy download URL returns HTTP 500.
Falls back to GitHub sample_datasets/*_sample.csv only for smoke tests.

Usage:
  python -m scripts.load_elephant --dataset oeq --n 10
  python -m scripts.load_elephant --dataset oeq --n 150  # requires full OSF data
"""
from __future__ import annotations

import argparse
import random
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

DATA_DIR = Path("data/elephant")
OSF_LEGACY_URL = (
    "https://osf.io/r3dmj/download/?view_only=37ee66a8020a45c29a38bd704ca61067"
)
OSF_API_FILES = (
    "https://api.osf.io/v2/nodes/r3dmj/files/osfstorage/"
    "?view_only=37ee66a8020a45c29a38bd704ca61067"
)
OSF_DATASETS_GUID = "4chzv"
OSF_API_DOWNLOAD = (
    f"https://osf.io/download/{OSF_DATASETS_GUID}/"
    "?view_only=37ee66a8020a45c29a38bd704ca61067"
)
GITHUB_SAMPLE = (
    "https://raw.githubusercontent.com/myracheng/elephant/main/sample_datasets/{fname}"
)

ELEPHANT_SEED = 44
MIN_FULL_ROWS = 50  # below this we treat the slice as sample-only

DATASET_FILES = {
    "oeq": ("OEQ.csv", "OEQ_sample.csv"),
    "aita_yta": ("AITA-YTA.csv", "AITA-YTA_sample.csv"),
    "ss": ("SS.csv", "SS_sample.csv"),
    "flip": ("AITA-NTA-FLIP.csv", "AITA-NTA-FLIP_sample.csv"),
    "og": ("AITA-NTA-OG.csv", "AITA-NTA-OG_sample.csv"),
}

METRICS_BY_DATASET = {
    "oeq": ["validation", "indirectness", "framing"],
    "aita_yta": ["validation", "indirectness", "framing"],
    "ss": ["framing"],
    "flip": ["moral"],
    "og": ["moral"],
}


class ElephantDataError(RuntimeError):
    """Raised when requested n exceeds available full-dataset rows."""


@dataclass
class ElephantItem:
    id: str
    dataset: str
    prompt: str
    human_response: str = ""
    human_scores: dict = field(default_factory=dict)
    pair_id: str = ""
    side: str = ""  # og | flip for moral pairs
    extra: dict = field(default_factory=dict)


def _is_sample_path(path: Path) -> bool:
    return path.name.endswith("_sample.csv")


def _zip_is_valid(zip_path: Path) -> bool:
    if not zip_path.exists() or zip_path.stat().st_size < 1_000_000:
        return False
    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            return any("OEQ.csv" in n for n in names)
    except zipfile.BadZipFile:
        return False


def _download_url(url: str, dest: Path, retries: int = 4) -> bool:
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=300, allow_redirects=True)
            if r.status_code == 200 and len(r.content) > 1_000_000:
                dest.write_bytes(r.content)
                print(f"  Saved {dest} ({dest.stat().st_size // 1024} KB)", flush=True)
                return True
            print(
                f"  Download attempt {attempt + 1}: HTTP {r.status_code}, "
                f"{len(r.content)} bytes from {url[:60]}",
                flush=True,
            )
        except Exception as e:
            print(f"  Download attempt {attempt + 1} failed: {e}", flush=True)
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    return False


def _download_osf_zip(dest: Path) -> bool:
    zip_path = dest / "datasets.zip"
    if _zip_is_valid(zip_path):
        return True
    if zip_path.exists():
        zip_path.unlink()

    dest.mkdir(parents=True, exist_ok=True)
    print("Downloading ELEPHANT datasets.zip", flush=True)

    # Prefer OSF file-GUID download (works when legacy node download returns 500).
    if _download_url(OSF_API_DOWNLOAD, zip_path):
        return True

    # Legacy node download.
    if _download_url(OSF_LEGACY_URL, zip_path):
        return True

    # Resolve download link from OSF API listing.
    try:
        r = requests.get(OSF_API_FILES, timeout=60)
        r.raise_for_status()
        for item in r.json().get("data", []):
            attrs = item.get("attributes", {})
            if attrs.get("name") != "datasets.zip":
                continue
            dl = item.get("links", {}).get("download", "")
            if dl and _download_url(dl, zip_path):
                return True
    except Exception as e:
        print(f"  OSF API listing failed: {e}", flush=True)

    if zip_path.exists():
        zip_path.unlink()
    return False


def _extract_csv(dataset_key: str, data_dir: Path) -> Optional[Path]:
    full_name, _sample_name = DATASET_FILES[dataset_key]
    local = data_dir / full_name
    if local.exists():
        return local
    zip_path = data_dir / "datasets.zip"
    if not zip_path.exists():
        if not _download_osf_zip(data_dir):
            return None
    try:
        with zipfile.ZipFile(zip_path) as zf:
            candidates = [n for n in zf.namelist() if n.endswith(full_name)]
            if not candidates:
                candidates = [n for n in zf.namelist() if full_name.lower() in n.lower()]
            if not candidates:
                print(f"  {full_name} not found in zip", flush=True)
                return None
            data_dir.mkdir(parents=True, exist_ok=True)
            local.write_bytes(zf.read(candidates[0]))
            print(f"  Extracted {local}", flush=True)
            return local
    except zipfile.BadZipFile as e:
        print(f"  Zip extract failed (bad zip): {e}", flush=True)
        zip_path.unlink(missing_ok=True)
        return None
    except Exception as e:
        print(f"  Zip extract failed: {e}", flush=True)
        return None


def _download_sample(sample_name: str, data_dir: Path) -> Path:
    dest = data_dir / sample_name
    if dest.exists():
        return dest
    data_dir.mkdir(parents=True, exist_ok=True)
    url = GITHUB_SAMPLE.format(fname=sample_name)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)
    print(f"  Downloaded sample {dest}", flush=True)
    return dest


def _resolve_csv(
    dataset_key: str,
    data_dir: Path = DATA_DIR,
    *,
    allow_sample: bool = False,
) -> Path:
    full_name, sample_name = DATASET_FILES[dataset_key]
    local = data_dir / full_name
    sample_local = data_dir / sample_name

    if local.exists():
        return local

    path = _extract_csv(dataset_key, data_dir)
    if path is not None:
        return path

    if allow_sample:
        if sample_local.exists():
            return sample_local
        return _download_sample(sample_name, data_dir)

    raise ElephantDataError(
        f"Full ELEPHANT dataset {full_name} unavailable. "
        f"OSF download failed; use --allow-sample only for smoke tests."
    )


def _row_count(dataset_key: str, data_dir: Path, allow_sample: bool) -> int:
    path = _resolve_csv(dataset_key, data_dir, allow_sample=allow_sample)
    return len(pd.read_csv(path))


def _validate_n(
    dataset: str,
    n: Optional[int],
    data_dir: Path,
    allow_sample: bool,
) -> None:
    if n is None or n <= 10:
        return
    if allow_sample:
        return

    if dataset == "flip_pairs":
        flip_n = _row_count("flip", data_dir, allow_sample=False)
        og_n = _row_count("og", data_dir, allow_sample=False)
        avail = min(flip_n, og_n)
        if avail < MIN_FULL_ROWS:
            raise ElephantDataError(
                f"FLIP/OG pair data has only {avail} rows; full OSF data required for n={n}."
            )
        if n > avail:
            print(
                f"  Warning: requested n={n} pairs but only {avail} available; using all.",
                flush=True,
            )
        return

    avail = _row_count(dataset, data_dir, allow_sample=False)
    if avail < MIN_FULL_ROWS:
        raise ElephantDataError(
            f"Dataset {dataset} has only {avail} rows ({_resolve_csv(dataset, data_dir, allow_sample=False).name}); "
            f"full OSF data required for n={n}. Run load_elephant with OSF access or use --allow-sample for smoke."
        )
    if n > avail:
        print(f"  Warning: requested n={n} but {dataset} has {avail} rows; using all.", flush=True)


def _row_id(row: pd.Series, idx: int) -> str:
    for col in ("id", "Unnamed: 0"):
        if col in row.index and pd.notna(row[col]):
            return str(row[col]).strip()
    return str(idx)


def _human_score_cols(row: pd.Series) -> dict:
    out = {}
    for m in ("validation", "indirectness", "framing"):
        col = f"{m}_human"
        if col in row.index and pd.notna(row[col]):
            try:
                out[m] = int(float(row[col]))
            except (TypeError, ValueError):
                pass
    return out


def load_elephant(
    dataset: str,
    n: Optional[int] = None,
    seed: int = ELEPHANT_SEED,
    data_dir: Path = DATA_DIR,
    *,
    allow_sample: bool = False,
    offset: int = 0,
) -> list[ElephantItem]:
    """Load stratified subsample of an ELEPHANT dataset."""
    dataset = dataset.lower().replace("-", "_")
    if dataset not in DATASET_FILES and dataset != "flip_pairs":
        raise ValueError(f"Unknown dataset {dataset}; choose from {list(DATASET_FILES)}")

    _validate_n(dataset, n, data_dir, allow_sample)

    if dataset == "flip_pairs":
        return _load_flip_pairs(
            n=n, seed=seed, data_dir=data_dir,
            allow_sample=allow_sample, offset=offset,
        )

    csv_path = _resolve_csv(dataset, data_dir, allow_sample=allow_sample)
    df = pd.read_csv(csv_path)
    if _is_sample_path(csv_path) and n is not None and n > len(df) and not allow_sample:
        raise ElephantDataError(
            f"Requested n={n} but only sample file {csv_path.name} ({len(df)} rows) is present."
        )
    if n is not None or offset > 0:
        rng = random.Random(seed)
        idxs = list(range(len(df)))
        rng.shuffle(idxs)
        if offset > 0:
            if offset >= len(idxs):
                raise ElephantDataError(
                    f"offset={offset} exceeds dataset {dataset} size {len(idxs)}"
                )
            idxs = idxs[offset:]
        take = n if n is not None else len(idxs)
        if take > len(idxs):
            print(
                f"  Warning: requested n={take} but only {len(idxs)} rows after "
                f"offset={offset}; using all.",
                flush=True,
            )
            take = len(idxs)
        df = df.iloc[idxs[:take]].reset_index(drop=True)

    items: list[ElephantItem] = []
    for i, row in df.iterrows():
        rid = _row_id(row, i)
        if dataset == "oeq":
            prompt = str(row.get("prompt", "") or "")
            human = str(row.get("human", "") or "")
        elif dataset == "aita_yta":
            prompt = str(row.get("prompt", "") or "")
            human = str(row.get("top_comment", "") or "")
        elif dataset == "ss":
            prompt = str(row.get("sentence", "") or "")
            human = ""
        elif dataset == "flip":
            prompt = str(row.get("prompt", "") or "")
            human = ""
        elif dataset == "og":
            prompt = str(row.get("prompt", "") or "")
            human = ""
        else:
            prompt = str(row.get("prompt", row.get("sentence", "")) or "")
            human = ""

        items.append(ElephantItem(
            id=rid,
            dataset=dataset,
            prompt=prompt.strip(),
            human_response=human.strip(),
            human_scores=_human_score_cols(row),
            pair_id=rid,
            extra={k: ("" if pd.isna(v) else str(v)) for k, v in row.items()
                   if k not in ("prompt", "human", "sentence", "top_comment")},
        ))
    return items


def _load_flip_pairs(
    n: Optional[int],
    seed: int,
    data_dir: Path,
    *,
    allow_sample: bool = False,
    offset: int = 0,
) -> list[ElephantItem]:
    """Return paired OG+FLIP items sharing pair_id for moral sycophancy."""
    flip_path = _resolve_csv("flip", data_dir, allow_sample=allow_sample)
    og_path = _resolve_csv("og", data_dir, allow_sample=allow_sample)
    df_flip = pd.read_csv(flip_path)
    df_og = pd.read_csv(og_path)
    n_pairs = min(len(df_flip), len(df_og))
    rng = random.Random(seed)
    idxs = list(range(min(len(df_flip), len(df_og))))
    rng.shuffle(idxs)
    if offset > 0:
        if offset >= len(idxs):
            raise ElephantDataError(
                f"offset={offset} exceeds flip pair count {len(idxs)}"
            )
        idxs = idxs[offset:]
    if n is not None:
        n_pairs = min(n, len(idxs))
    else:
        n_pairs = len(idxs)
    idxs = idxs[:n_pairs]

    items: list[ElephantItem] = []
    for j, idx in enumerate(idxs):
        pair_id = f"pair_{j:04d}"
        row_og = df_og.iloc[idx]
        row_flip = df_flip.iloc[idx]
        items.append(ElephantItem(
            id=_row_id(row_og, idx) + "_og",
            dataset="flip_pairs",
            prompt=str(row_og.get("prompt", "") or "").strip(),
            pair_id=pair_id,
            side="og",
        ))
        items.append(ElephantItem(
            id=_row_id(row_flip, idx) + "_flip",
            dataset="flip_pairs",
            prompt=str(row_flip.get("prompt", "") or "").strip(),
            pair_id=pair_id,
            side="flip",
        ))
    return items


def metrics_for_dataset(dataset: str) -> list[str]:
    dataset = dataset.lower().replace("-", "_")
    if dataset == "flip_pairs":
        return ["moral"]
    return METRICS_BY_DATASET.get(dataset, ["validation", "indirectness", "framing"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="oeq")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=ELEPHANT_SEED)
    ap.add_argument("--allow-sample", action="store_true")
    ap.add_argument("--download-only", action="store_true")
    args = ap.parse_args()

    if args.download_only:
        ok = _download_osf_zip(DATA_DIR)
        print("download ok" if ok else "download failed")
        return 0 if ok else 1

    ds = args.dataset.lower()
    if ds == "flip_pairs":
        items = _load_flip_pairs(
            n=args.n, seed=args.seed, data_dir=DATA_DIR, allow_sample=args.allow_sample,
        )
    else:
        items = load_elephant(
            ds, n=args.n, seed=args.seed, allow_sample=args.allow_sample,
        )
    print(f"{ds}: {len(items)} items")
    if items:
        print(f"  sample id={items[0].id} prompt_len={len(items[0].prompt)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
