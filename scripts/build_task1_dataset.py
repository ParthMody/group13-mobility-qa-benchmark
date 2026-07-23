#!/usr/bin/env python3
"""Combine approved Task 1 source files into one normalized dataset."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mobility_qa.task1_split import (  # noqa: E402
    deduplicate_task1_records,
    load_task1_records,
    validate_task1_records,
    write_task1_split,
)


DEFAULT_JSONL = REPO_ROOT / "data/processed/task1_all_records.jsonl"
DEFAULT_CSV = REPO_ROOT / "data/processed/task1_all_records.csv"
DEFAULT_REJECTIONS = (
    REPO_ROOT / "data/benchmark/task1/task1_rejected_records.csv"
)
EXCLUDED_FILENAMES = {
    DEFAULT_JSONL.name,
    DEFAULT_CSV.name,
    "task1_initial_400.jsonl",
    "task1_initial_400.csv",
    "task1_extended.jsonl",
    "task1_extended.csv",
}


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def discover_task1_source_files() -> list[Path]:
    """Find only approved Task 1 CSV and JSONL source files."""
    paths: set[Path] = set()
    for source_dir in (
        REPO_ROOT / "data/examples",
        REPO_ROOT / "data/processed",
    ):
        if not source_dir.exists():
            continue
        for pattern in ("task1_*.jsonl", "task1_*.csv"):
            for path in source_dir.glob(pattern):
                if path.name not in EXCLUDED_FILENAMES:
                    paths.add(path)
    return sorted(paths, key=lambda path: str(path.relative_to(REPO_ROOT)))


def _write_rejections(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["question_id", "source_file", "rejection_reason"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "question_id": row.get("question_id", ""),
                    "source_file": row.get("source_file", ""),
                    "rejection_reason": row.get("rejection_reason", ""),
                }
            )


def build_task1_dataset(
    output_csv: Path = DEFAULT_CSV,
    output_jsonl: Path = DEFAULT_JSONL,
    rejection_report: Path = DEFAULT_REJECTIONS,
) -> dict[str, object]:
    """Load, validate, deduplicate, and write all approved Task 1 records."""
    source_paths = discover_task1_source_files()
    loaded: list[dict[str, object]] = []

    for source_path in source_paths:
        source_label = str(source_path.relative_to(REPO_ROOT))
        for record in load_task1_records(source_path):
            record["_source_file"] = source_label
            loaded.append(record)

    valid, invalid = validate_task1_records(loaded)
    unique, duplicates = deduplicate_task1_records(valid)

    rejection_rows = [
        {
            "question_id": row.get("question_id", ""),
            "source_file": row.get("source_file", ""),
            "rejection_reason": row.get("rejection_reason", ""),
        }
        for row in invalid
    ]
    rejection_rows.extend(
        {
            "question_id": record.get("question_id", ""),
            "source_file": record.get("_source_file", ""),
            "rejection_reason": (
                "duplicate: identical question_id and normalized record already included"
            ),
        }
        for record in duplicates
    )

    write_task1_split(unique, output_csv, output_jsonl)
    _write_rejections(rejection_rows, rejection_report)

    return {
        "source_paths": [
            str(path.relative_to(REPO_ROOT)) for path in source_paths
        ],
        "records_loaded": len(loaded),
        "valid_records": len(unique),
        "invalid_records_excluded": len(invalid),
        "duplicate_records_removed": len(duplicates),
        "output_csv": _display_path(output_csv),
        "output_jsonl": _display_path(output_jsonl),
        "rejection_report": _display_path(rejection_report),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the normalized Task 1 dataset from approved sources."
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument(
        "--rejection-report", type=Path, default=DEFAULT_REJECTIONS
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = build_task1_dataset(
            output_csv=args.output_csv,
            output_jsonl=args.output_jsonl,
            rejection_report=args.rejection_report,
        )
    except ValueError as exc:
        print(f"Task 1 dataset build failed: {exc}", file=sys.stderr)
        return 1

    print("Task 1 combined dataset built")
    print(f"Source files: {len(summary['source_paths'])}")
    print(f"Records loaded: {summary['records_loaded']}")
    print(f"Valid unique records: {summary['valid_records']}")
    print(f"Invalid records excluded: {summary['invalid_records_excluded']}")
    print(f"Duplicate records removed: {summary['duplicate_records_removed']}")
    print(f"CSV: {summary['output_csv']}")
    print(f"JSONL: {summary['output_jsonl']}")
    print(f"Rejections: {summary['rejection_report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
