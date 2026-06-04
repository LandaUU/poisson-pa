#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import requests


@dataclass(frozen=True, slots=True)
class SnapDataset:
    key: str
    filename: str
    url: str
    description: str
    default: bool = True


DATASETS: tuple[SnapDataset, ...] = (
    SnapDataset(
        key="college-msg",
        filename="CollegeMsg.txt.gz",
        url="https://snap.stanford.edu/data/CollegeMsg.txt.gz",
        description="CollegeMsg temporal network",
    ),
    SnapDataset(
        key="email-eu-core-temporal",
        filename="email-Eu-core-temporal.txt.gz",
        url="https://snap.stanford.edu/data/email-Eu-core-temporal.txt.gz",
        description="email-Eu-core temporal network",
    ),
    SnapDataset(
        key="mathoverflow-a2q",
        filename="sx-mathoverflow-a2q.txt.gz",
        url="https://snap.stanford.edu/data/sx-mathoverflow-a2q.txt.gz",
        description="MathOverflow answers-to-questions temporal network",
    ),
    SnapDataset(
        key="email-eu-core-temporal-dept1",
        filename="email-Eu-core-temporal-Dept1.txt.gz",
        url="https://snap.stanford.edu/data/email-Eu-core-temporal-Dept1.txt.gz",
        description="email-Eu-core temporal Department 1 sub-network",
        default=False,
    ),
    SnapDataset(
        key="email-eu-core-temporal-dept2",
        filename="email-Eu-core-temporal-Dept2.txt.gz",
        url="https://snap.stanford.edu/data/email-Eu-core-temporal-Dept2.txt.gz",
        description="email-Eu-core temporal Department 2 sub-network",
        default=False,
    ),
    SnapDataset(
        key="email-eu-core-temporal-dept3",
        filename="email-Eu-core-temporal-Dept3.txt.gz",
        url="https://snap.stanford.edu/data/email-Eu-core-temporal-Dept3.txt.gz",
        description="email-Eu-core temporal Department 3 sub-network",
        default=False,
    ),
    SnapDataset(
        key="email-eu-core-temporal-dept4",
        filename="email-Eu-core-temporal-Dept4.txt.gz",
        url="https://snap.stanford.edu/data/email-Eu-core-temporal-Dept4.txt.gz",
        description="email-Eu-core temporal Department 4 sub-network",
        default=False,
    ),
)


def parse_args() -> argparse.Namespace:
    dataset_keys = sorted(dataset.key for dataset in DATASETS)
    parser = argparse.ArgumentParser(description="Download SNAP temporal edge-list datasets used in the project.")
    parser.add_argument("--output-dir", default="data", help="Directory where downloaded files will be saved.")
    parser.add_argument("--force", action="store_true", help="Download files even if they already exist locally.")
    parser.add_argument(
        "--all", action="store_true", help="Download default datasets and optional email department files."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=dataset_keys,
        help="Dataset key to download. Can be passed multiple times. Overrides the default set.",
    )
    return parser.parse_args()


def selected_datasets(args: argparse.Namespace) -> list[SnapDataset]:
    if args.dataset:
        requested = set(args.dataset)
        return [dataset for dataset in DATASETS if dataset.key in requested]
    if args.all:
        return list(DATASETS)
    return [dataset for dataset in DATASETS if dataset.default]


def validate_temporal_gzip(path: Path, *, sample_lines: int = 10) -> None:
    checked = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 3:
                raise ValueError(f"Expected at least three columns at {path}:{line_no}: {raw_line!r}")
            try:
                int(parts[0])
                int(parts[1])
                int(parts[2])
            except ValueError as exc:
                raise ValueError(f"Invalid integer columns at {path}:{line_no}: {raw_line!r}") from exc
            checked += 1
            if checked >= sample_lines:
                return

    if checked == 0:
        raise ValueError(f"No temporal edges found in {path}")


def download_dataset(session: requests.Session, dataset: SnapDataset, *, output_dir: Path, force: bool) -> Path:
    output_path = output_dir / dataset.filename
    part_path = output_path.with_suffix(output_path.suffix + ".part")

    if output_path.exists() and not force:
        validate_temporal_gzip(output_path)
        print(f"OK existing: {output_path}")
        return output_path

    if part_path.exists():
        part_path.unlink()

    print(f"Downloading {dataset.key}: {dataset.url}")
    try:
        with session.get(dataset.url, stream=True, timeout=(10, 120)) as response:
            response.raise_for_status()
            with part_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)

        validate_temporal_gzip(part_path)
        part_path.replace(output_path)
    except Exception:
        if part_path.exists():
            part_path.unlink()
        raise

    print(f"Saved: {output_path}")
    return output_path


def download_all(datasets: Sequence[SnapDataset], *, output_dir: Path, force: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "master-deg-project SNAP downloader"})

    for dataset in datasets:
        download_dataset(session, dataset, output_dir=output_dir, force=force)


def main() -> None:
    args = parse_args()
    datasets = selected_datasets(args)
    if not datasets:
        raise RuntimeError("No datasets selected")
    download_all(datasets, output_dir=Path(args.output_dir), force=args.force)


if __name__ == "__main__":
    main()
