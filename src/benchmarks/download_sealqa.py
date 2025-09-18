"""Download the full SealQA dataset (all subsets and available splits) to local files.

This utility fetches all configured subsets of the Hugging Face dataset
`vtllms/sealqa` (seal_0, seal_hard, longseal) and writes them to the
`data/` directory in either Parquet or JSONL format.

Usage (via uv):
  uv run python src/benchmarks/download_sealqa.py --out data/ --format parquet

Reference dataset card:
  - https://huggingface.co/datasets/vtllms/sealqa
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import List

from datasets import Dataset, DatasetDict, load_dataset


DEFAULT_SUBSETS: List[str] = ["seal_0", "seal_hard", "longseal"]
DATASET_NAME: str = "vtllms/sealqa"


def ensure_directory(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def write_parquet(ds: Dataset, file_path: str) -> None:
    ds.to_parquet(file_path)


def write_jsonl(ds: Dataset, file_path: str) -> None:
    # Use pandas for robust JSON Lines export with UTF-8
    df = ds.to_pandas()
    df.to_json(file_path, orient="records", lines=True, force_ascii=False)


def save_metadata(base_dir: str, subset: str, splits: List[str]) -> None:
    ensure_directory(base_dir)
    meta_path = os.path.join(base_dir, "metadata.json")
    metadata = {
        "dataset": DATASET_NAME,
        "subset": subset,
        "splits": splits,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def download_subset(subset: str, out_root: str, file_format: str) -> None:
    print(f"Downloading subset '{subset}' from {DATASET_NAME} ...")
    dataset: DatasetDict | Dataset = load_dataset(DATASET_NAME, name=subset)

    out_dir_for_subset = os.path.join(out_root, "sealqa", subset)
    ensure_directory(out_dir_for_subset)

    if isinstance(dataset, DatasetDict):
        available_splits = list(dataset.keys())
        for split_name, split_ds in dataset.items():
            file_name = f"{split_name}.{file_format}"
            file_path = os.path.join(out_dir_for_subset, file_name)
            if file_format == "parquet":
                write_parquet(split_ds, file_path)
            elif file_format == "jsonl":
                write_jsonl(split_ds, file_path)
            else:
                raise ValueError(f"Unsupported format: {file_format}")
            print(f"  Saved: {file_path}")
        save_metadata(out_dir_for_subset, subset=subset, splits=available_splits)
    else:
        # Single Dataset (rare for this dataset, but handle gracefully)
        file_name = f"all.{file_format}"
        file_path = os.path.join(out_dir_for_subset, file_name)
        if file_format == "parquet":
            write_parquet(dataset, file_path)
        elif file_format == "jsonl":
            write_jsonl(dataset, file_path)
        else:
            raise ValueError(f"Unsupported format: {file_format}")
        print(f"  Saved: {file_path}")
        save_metadata(out_dir_for_subset, subset=subset, splits=["all"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the full SealQA dataset (all subsets and available splits)"
        )
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data",
        help="Output root directory (default: data)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["parquet", "jsonl"],
        default="parquet",
        help="File format for saved datasets (default: parquet)",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        default="all",
        help=(
            "Comma-separated list of subsets to download (seal_0,seal_hard,longseal) "
            "or 'all' (default)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root: str = args.out
    file_format: str = args.format

    if args.subsets == "all":
        subsets: List[str] = DEFAULT_SUBSETS
    else:
        subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]
        unknown = [s for s in subsets if s not in DEFAULT_SUBSETS]
        if unknown:
            raise ValueError(
                f"Unknown subset(s): {unknown}. Allowed: {', '.join(DEFAULT_SUBSETS)}"
            )

    for subset in subsets:
        download_subset(subset=subset, out_root=out_root, file_format=file_format)

    print("Done.")


if __name__ == "__main__":
    main()


