#!/usr/bin/env python3
"""Create deterministic initial and extended splits for Task 1."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mobility_qa.task1_split import (  # noqa: E402
    create_task1_initial_extended_split,
    create_task1_split_manifest,
    deduplicate_task1_records,
    load_task1_records,
    validate_task1_records,
    write_task1_split,
)


DEFAULT_INPUT = REPO_ROOT / "data/processed/task1_all_records.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/benchmark/task1"
DEFAULT_INITIAL_CSV = DEFAULT_OUTPUT_DIR / "task1_initial_400.csv"
DEFAULT_INITIAL_JSONL = DEFAULT_OUTPUT_DIR / "task1_initial_400.jsonl"
DEFAULT_EXTENDED_CSV = DEFAULT_OUTPUT_DIR / "task1_extended.csv"
DEFAULT_EXTENDED_JSONL = DEFAULT_OUTPUT_DIR / "task1_extended.jsonl"
DEFAULT_MANIFEST = DEFAULT_OUTPUT_DIR / "task1_split_manifest.json"
DEFAULT_REJECTIONS = DEFAULT_OUTPUT_DIR / "task1_rejected_records.csv"
EXCLUDED_SOURCE_FILENAMES = {
    "task1_all_records.jsonl",
    "task1_all_records.csv",
    "task1_initial_400.jsonl",
    "task1_initial_400.csv",
    "task1_extended.jsonl",
    "task1_extended.csv",
}


def _read_existing_rejections(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _manifest_source_paths(input_path: Path) -> list[Path]:
    if input_path.resolve() != DEFAULT_INPUT.resolve():
        return [input_path]

    sources: set[Path] = set()
    for source_dir in (
        REPO_ROOT / "data/examples",
        REPO_ROOT / "data/processed",
    ):
        if not source_dir.exists():
            continue
        for pattern in ("task1_*.jsonl", "task1_*.csv"):
            sources.update(
                path
                for path in source_dir.glob(pattern)
                if path.name not in EXCLUDED_SOURCE_FILENAMES
            )
    return [
        path.relative_to(REPO_ROOT)
        for path in sorted(
            sources, key=lambda path: str(path.relative_to(REPO_ROOT))
        )
    ]


def _format_counts(counts: object) -> str:
    if not isinstance(counts, dict) or not counts:
        return "(none)"
    return ", ".join(f"{key}={value}" for key, value in counts.items())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Task 1 initial and extended benchmark splits."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--initial-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=5925)
    parser.add_argument("--initial-csv", type=Path, default=DEFAULT_INITIAL_CSV)
    parser.add_argument(
        "--initial-jsonl", type=Path, default=DEFAULT_INITIAL_JSONL
    )
    parser.add_argument("--extended-csv", type=Path, default=DEFAULT_EXTENDED_CSV)
    parser.add_argument(
        "--extended-jsonl", type=Path, default=DEFAULT_EXTENDED_JSONL
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--rejection-report", type=Path, default=DEFAULT_REJECTIONS
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        loaded = load_task1_records(args.input)
        for record in loaded:
            record["_source_file"] = str(args.input)
        valid, invalid = validate_task1_records(loaded)
        unique, duplicates = deduplicate_task1_records(valid)
        initial, extended = create_task1_initial_extended_split(
            unique,
            initial_size=args.initial_size,
            seed=args.seed,
        )
    except (OSError, ValueError) as exc:
        print(f"Task 1 split creation failed: {exc}", file=sys.stderr)
        return 1

    existing_rejections: list[dict[str, object]] = []
    if args.input.resolve() == DEFAULT_INPUT.resolve():
        existing_rejections.extend(_read_existing_rejections(args.rejection_report))

    new_rejections: list[dict[str, object]] = [
        {
            "question_id": row.get("question_id", ""),
            "source_file": row.get("source_file", str(args.input)),
            "rejection_reason": row.get("rejection_reason", ""),
        }
        for row in invalid
    ]
    new_rejections.extend(
        {
            "question_id": record.get("question_id", ""),
            "source_file": record.get("_source_file", str(args.input)),
            "rejection_reason": (
                "duplicate: identical question_id and normalized record already included"
            ),
        }
        for record in duplicates
    )
    all_rejections = existing_rejections + new_rejections
    _write_rejections(all_rejections, args.rejection_report)

    duplicate_count = sum(
        str(row.get("rejection_reason", "")).startswith("duplicate:")
        for row in all_rejections
    )
    invalid_count = len(all_rejections) - duplicate_count

    write_task1_split(initial, args.initial_csv, args.initial_jsonl)
    write_task1_split(extended, args.extended_csv, args.extended_jsonl)
    manifest = create_task1_split_manifest(
        initial,
        extended,
        source_paths=_manifest_source_paths(args.input),
        initial_size=args.initial_size,
        seed=args.seed,
        duplicate_records_removed=duplicate_count,
        invalid_records_excluded=invalid_count,
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Total valid Task 1 records: {manifest['total_valid_records']}")
    print(f"Initial records: {manifest['initial_records']}")
    print(f"Extended records: {manifest['extended_records']}")
    print(f"Excluded invalid records: {manifest['invalid_records_excluded']}")
    print(f"Removed duplicate records: {manifest['duplicate_records_removed']}")
    for label, manifest_key in (
        ("City distribution (total)", "city_counts_total"),
        ("City distribution (initial)", "city_counts_initial"),
        ("City distribution (extended)", "city_counts_extended"),
        ("Difficulty distribution (total)", "difficulty_counts_total"),
        ("Difficulty distribution (initial)", "difficulty_counts_initial"),
        ("Difficulty distribution (extended)", "difficulty_counts_extended"),
        ("Answer-category distribution (total)", "answer_category_counts_total"),
        (
            "Answer-category distribution (initial)",
            "answer_category_counts_initial",
        ),
        (
            "Answer-category distribution (extended)",
            "answer_category_counts_extended",
        ),
    ):
        print(f"{label}: {_format_counts(manifest[manifest_key])}")
    print(f"Initial CSV: {args.initial_csv}")
    print(f"Initial JSONL: {args.initial_jsonl}")
    print(f"Extended CSV: {args.extended_csv}")
    print(f"Extended JSONL: {args.extended_jsonl}")
    print(f"Manifest: {args.manifest}")
    print(f"Rejections: {args.rejection_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
