"""
scripts/load_elephant.py -- ELEPHANT benchmark data loader.

Downloads and caches OSF datasets.zip into data/elephant/ (gitignored).
Falls back to GitHub sample_datasets/*_sample.csv when full data absent.

Usage:
  python -m scripts.load_elephant --dataset oeq --n 10
"""
from __future__ import annotations

import argparse
import io
import random
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

DATA_DIR = Path("data/elephant")
OSF_URL = (
    "https://osf.io/r3dmj/download/?view_only=37ee66a8020a45c29a38bd704ca61067"
)
GITHUB_SAMPLE = (
    "https://raw.githubusercontent.com/myracheng/elephant/main/sample_datasets/{fname}"
)

ELEPHANT_SEED = 44

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


def _download_osf_zip(dest: Path) -> bool:
    zip_path = dest / "datasets.zip"
    if zip_path.exists():
        if zip_path.stat().st_size > 1_000_000:
            return True
        zip_path.unlink()  # remove corrupt stub from failed OSF download
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading ELEPHANT datasets from OSF -> {zip_path}", flush=True)
    try:
        r = requests.get(OSF_URL, timeout=300, allow_redirects=True)
        r.raise_for_status()
        zip_path.write_bytes(r.content)
        print(f"  Saved {zip_path.stat().st_size // 1024} KB", flush=True)
        return True
    except Exception as e:
        print(f"  OSF download failed: {e}", flush=True)
        return False


def _extract_csv(dataset_key: str, data_dir: Path) -> Optional[Path]:
    full_name, sample_name = DATASET_FILES[dataset_key]
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
            content = zf.read(candidates[0])
            local.write_bytes(content)
            print(f"  Extracted {local}", flush=True)
            return local
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


def _resolve_csv(dataset_key: str, data_dir: Path = DATA_DIR) -> Path:
    full_name, sample_name = DATASET_FILES[dataset_key]
    local = data_dir / full_name
    sample_local = data_dir / sample_name
    if local.exists():
        return local
    if sample_local.exists():
        return sample_local
    path = _extract_csv(dataset_key, data_dir)
    if path is not None:
        return path
    return _download_sample(sample_name, data_dir)


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
) -> list[ElephantItem]:
    """Load stratified subsample of an ELEPHANT dataset."""
    dataset = dataset.lower().replace("-", "_")
    if dataset not in DATASET_FILES and dataset != "flip_pairs":
        raise ValueError(f"Unknown dataset {dataset}; choose from {list(DATASET_FILES)}")

    if dataset == "flip_pairs":
        return _load_flip_pairs(n=n, seed=seed, data_dir=data_dir)

    csv_path = _resolve_csv(dataset, data_dir)
    df = pd.read_csv(csv_path)
    if n is not None:
        if n > len(df):
            print(f"  Warning: requested n={n} but {dataset} has {len(df)} rows; using all.", flush=True)
            n = len(df)
        if n < len(df):
            rng = random.Random(seed)
            idxs = list(range(len(df)))
            rng.shuffle(idxs)
            df = df.iloc[idxs[:n]].reset_index(drop=True)

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
) -> list[ElephantItem]:
    """Return paired OG+FLIP items sharing pair_id for moral sycophancy."""
    flip_path = _resolve_csv("flip", data_dir)
    og_path = _resolve_csv("og", data_dir)
    df_flip = pd.read_csv(flip_path)
    df_og = pd.read_csv(og_path)
    n_pairs = min(len(df_flip), len(df_og))
    if n is not None:
        n_pairs = min(n, n_pairs)
    rng = random.Random(seed)
    idxs = list(range(min(len(df_flip), len(df_og))))
    rng.shuffle(idxs)
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
    args = ap.parse_args()
    ds = args.dataset.lower()
    if ds == "flip_pairs":
        items = _load_flip_pairs(n=args.n, seed=args.seed, data_dir=DATA_DIR)
    else:
        items = load_elephant(ds, n=args.n, seed=args.seed)
    print(f"{ds}: {len(items)} items")
    if items:
        print(f"  sample id={items[0].id} prompt_len={len(items[0].prompt)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
