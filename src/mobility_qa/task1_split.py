"""Task 1 dataset normalization, validation, and split helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK1_NAME = "task1_next_poi_category"
TASK1_ANSWER_TYPE = "multiple choice"
VALID_DIFFICULTIES = {"easy", "medium", "hard"}
DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}

TASK1_FIELDS = [
    "question_id",
    "task",
    "city",
    "context",
    "answer_type",
    "options",
    "correct_answer",
    "reasoning",
    "difficulty",
    "source_dataset",
    "verification_status",
]


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _parse_json(value: object, default: object) -> object:
    if not isinstance(value, str) or not value.strip():
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _metadata(record: Mapping[str, object]) -> Mapping[str, object]:
    value = record.get("metadata", {})
    if isinstance(value, Mapping):
        return value
    decoded = _parse_json(value, {})
    return decoded if isinstance(decoded, Mapping) else {}


def _normalise_context(record: Mapping[str, object]) -> str:
    for field in ("context", "context_text", "history"):
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (list, dict)) and value:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)

    sequence = record.get("context_sequence")
    if isinstance(sequence, str):
        decoded = _parse_json(sequence, sequence)
        sequence = decoded
    if isinstance(sequence, (list, dict)) and sequence:
        return json.dumps(sequence, ensure_ascii=False, sort_keys=True)
    return ""


def _normalise_options(value: object) -> list[str]:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        decoded = _parse_json(stripped, None)
        if isinstance(decoded, list):
            value = decoded
        else:
            return [part.strip() for part in stripped.split("|") if part.strip()]

    if isinstance(value, (list, tuple)):
        return [str(option).strip() for option in value if str(option).strip()]
    return []


def _normalise_answer_type(
    value: object, metadata: Mapping[str, object], options: list[str]
) -> str:
    raw = _text(value) or _text(metadata.get("answer_type"))
    compact = raw.casefold().replace("_", " ").replace("-", " ")
    compact = " ".join(compact.split())
    if compact in {"closed", "multiple choice", "mcq"}:
        return TASK1_ANSWER_TYPE
    if compact in {"open", "written"}:
        return "written"
    if not compact and options:
        return TASK1_ANSWER_TYPE
    return compact


def _normalise_task1_record(record: Mapping[str, object]) -> dict[str, Any]:
    metadata = _metadata(record)
    options = _normalise_options(record.get("options", record.get("choices", [])))
    difficulty = _text(record.get("difficulty")) or _text(metadata.get("difficulty"))
    verification_status = _text(record.get("verification_status")) or _text(
        metadata.get("verification_status")
    )

    normalised: dict[str, Any] = {
        "question_id": _text(record.get("question_id")),
        "task": _text(record.get("task")),
        "city": _text(record.get("city")),
        "context": _normalise_context(record),
        "answer_type": _normalise_answer_type(
            record.get("answer_type"), metadata, options
        ),
        "options": options,
        "correct_answer": _text(
            record.get("correct_answer", record.get("answer"))
        ),
        "reasoning": _text(record.get("reasoning", record.get("rationale"))),
        "difficulty": difficulty.casefold(),
        "source_dataset": _text(record.get("source_dataset")),
        "verification_status": verification_status,
    }

    for key, value in record.items():
        if str(key).startswith("_"):
            normalised[str(key)] = value
    return normalised


def _public_record(record: Mapping[str, object]) -> dict[str, Any]:
    return {field: record.get(field, "") for field in TASK1_FIELDS}


def load_task1_records(path: str | Path) -> list[dict[str, Any]]:
    """Load and normalize Task 1 records from CSV or JSONL."""
    input_path = Path(path)
    suffix = input_path.suffix.casefold()
    raw_records: list[Mapping[str, object]] = []

    if suffix == ".jsonl":
        with input_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{input_path}: invalid JSON on line {line_number}: {exc}"
                    ) from exc
                if not isinstance(value, Mapping):
                    raise ValueError(
                        f"{input_path}: line {line_number} is not a JSON object"
                    )
                raw_records.append(value)
    elif suffix == ".csv":
        with input_path.open("r", encoding="utf-8", newline="") as handle:
            raw_records.extend(dict(row) for row in csv.DictReader(handle))
    else:
        raise ValueError(f"Unsupported Task 1 input format: {input_path}")

    return [_normalise_task1_record(record) for record in raw_records]


def _validation_errors(record: Mapping[str, object]) -> list[str]:
    errors: list[str] = []
    if not _text(record.get("question_id")):
        errors.append("question_id is required")
    if record.get("task") != TASK1_NAME:
        errors.append(f"task must equal {TASK1_NAME}")
    if not _text(record.get("city")):
        errors.append("city is required")
    if not _text(record.get("context")):
        errors.append("context is required")
    if record.get("answer_type") != TASK1_ANSWER_TYPE:
        errors.append("answer_type must equal multiple choice")

    options = record.get("options")
    if not isinstance(options, list) or not options:
        errors.append("options must be a non-empty pipe-separated value or list")
        options = []
    elif not all(isinstance(option, str) and option.strip() for option in options):
        errors.append("options must contain non-empty text values")

    correct_answer = _text(record.get("correct_answer"))
    if not correct_answer:
        errors.append("correct_answer is required")
    elif correct_answer not in options:
        errors.append("correct_answer must appear exactly in options")

    if record.get("difficulty") not in VALID_DIFFICULTIES:
        errors.append("difficulty must be easy, medium, or hard")
    if not _text(record.get("source_dataset")):
        errors.append("source_dataset is required")
    if not _text(record.get("reasoning")):
        errors.append("reasoning is required")
    return errors


def validate_task1_records(
    records: Iterable[Mapping[str, object]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return valid normalized records and rejected records with reasons."""
    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for record in records:
        normalised = _normalise_task1_record(record)
        errors = _validation_errors(normalised)
        if errors:
            rejected.append(
                {
                    "question_id": normalised.get("question_id", ""),
                    "source_file": normalised.get("_source_file", ""),
                    "rejection_reason": "; ".join(errors),
                    "record": _public_record(normalised),
                }
            )
        else:
            valid.append(normalised)
    return valid, rejected


def _record_fingerprint(record: Mapping[str, object]) -> str:
    payload = json.dumps(
        _public_record(record),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def deduplicate_task1_records(
    records: Iterable[Mapping[str, object]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove identical duplicate IDs and reject conflicting duplicate IDs."""
    unique_by_id: dict[str, dict[str, Any]] = {}
    fingerprint_by_id: dict[str, str] = {}
    duplicates: list[dict[str, Any]] = []

    for record in records:
        normalised = _normalise_task1_record(record)
        question_id = normalised["question_id"]
        if not question_id:
            raise ValueError("Cannot deduplicate a Task 1 record without question_id")

        fingerprint = _record_fingerprint(normalised)
        if question_id not in unique_by_id:
            unique_by_id[question_id] = normalised
            fingerprint_by_id[question_id] = fingerprint
            continue

        if fingerprint_by_id[question_id] != fingerprint:
            first_source = unique_by_id[question_id].get("_source_file", "unknown")
            second_source = normalised.get("_source_file", "unknown")
            raise ValueError(
                "Conflicting duplicate question_id "
                f"'{question_id}' in {first_source} and {second_source}"
            )
        duplicates.append(normalised)

    return list(unique_by_id.values()), duplicates


def _readable_sort_key(record: Mapping[str, object]) -> tuple[object, ...]:
    return (
        _text(record.get("city")).casefold(),
        DIFFICULTY_ORDER.get(_text(record.get("difficulty")).casefold(), 99),
        _text(record.get("correct_answer")).casefold(),
        _text(record.get("question_id")).casefold(),
    )


def _seeded_ranks(
    records: list[dict[str, Any]], seed: int
) -> dict[str, int]:
    shuffled = sorted(records, key=_readable_sort_key)
    random.Random(seed).shuffle(shuffled)
    return {record["question_id"]: index for index, record in enumerate(shuffled)}


def create_task1_initial_extended_split(
    records: Iterable[Mapping[str, object]],
    initial_size: int = 400,
    seed: int = 5925,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create a deterministic, stratified initial and extended Task 1 split."""
    if initial_size < 0:
        raise ValueError("initial_size must be zero or greater")

    valid, rejected = validate_task1_records(records)
    if rejected:
        raise ValueError(
            "Split input contains invalid Task 1 records; validate and exclude them first"
        )
    unique, _ = deduplicate_task1_records(valid)
    unique = [_public_record(record) for record in unique]

    if len(unique) <= initial_size:
        return (
            [_public_record(record) for record in sorted(unique, key=_readable_sort_key)],
            [],
        )
    if initial_size == 0:
        return [], [_public_record(record) for record in sorted(unique, key=_readable_sort_key)]

    ranks = _seeded_ranks(unique, seed)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    city_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    answer_counts: Counter[str] = Counter()
    joint_counts: Counter[tuple[str, str, str]] = Counter()

    def add(record: dict[str, Any]) -> None:
        selected.append(record)
        selected_ids.add(record["question_id"])
        city_counts[record["city"]] += 1
        difficulty_counts[record["difficulty"]] += 1
        answer_counts[record["correct_answer"]] += 1
        joint_counts[
            (record["city"], record["difficulty"], record["correct_answer"])
        ] += 1

    def score(record: Mapping[str, object]) -> tuple[int, int, int, int, int]:
        joint = (
            str(record["city"]),
            str(record["difficulty"]),
            str(record["correct_answer"]),
        )
        return (
            city_counts[str(record["city"])],
            difficulty_counts[str(record["difficulty"])],
            answer_counts[str(record["correct_answer"])],
            joint_counts[joint],
            ranks[str(record["question_id"])],
        )

    for field in ("city", "difficulty", "correct_answer"):
        values = sorted({str(record[field]) for record in unique}, key=str.casefold)
        counts = {
            "city": city_counts,
            "difficulty": difficulty_counts,
            "correct_answer": answer_counts,
        }[field]
        for value in values:
            if len(selected) >= initial_size:
                break
            if counts[value]:
                continue
            candidates = [
                record
                for record in unique
                if record["question_id"] not in selected_ids
                and str(record[field]) == value
            ]
            if candidates:
                add(min(candidates, key=score))

    while len(selected) < initial_size:
        remaining = [
            record for record in unique if record["question_id"] not in selected_ids
        ]
        add(min(remaining, key=score))

    extended = [
        _public_record(record)
        for record in sorted(unique, key=_readable_sort_key)
        if record["question_id"] not in selected_ids
    ]
    return [_public_record(record) for record in selected], extended


def write_task1_split(
    records: Iterable[Mapping[str, object]],
    csv_path: str | Path,
    jsonl_path: str | Path,
) -> None:
    """Write equivalent Task 1 CSV and JSONL files."""
    output_records = [
        _public_record(_normalise_task1_record(record)) for record in records
    ]
    csv_output = Path(csv_path)
    jsonl_output = Path(jsonl_path)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    jsonl_output.parent.mkdir(parents=True, exist_ok=True)

    with csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TASK1_FIELDS)
        writer.writeheader()
        for record in output_records:
            row = dict(record)
            row["options"] = " | ".join(record["options"])
            writer.writerow(row)

    with jsonl_output.open("w", encoding="utf-8") as handle:
        for record in output_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _counts(records: Iterable[Mapping[str, object]], field: str) -> dict[str, int]:
    return dict(
        sorted(
            Counter(str(record.get(field, "")) for record in records).items(),
            key=lambda item: item[0].casefold(),
        )
    )


def create_task1_split_manifest(
    initial_records: Iterable[Mapping[str, object]],
    extended_records: Iterable[Mapping[str, object]],
    source_paths: Iterable[str | Path],
    initial_size: int = 400,
    seed: int = 5925,
    duplicate_records_removed: int = 0,
    invalid_records_excluded: int = 0,
    created_at: str | None = None,
    notes: Iterable[str] | None = None,
) -> dict[str, object]:
    """Build the Task 1 split manifest and distribution summaries."""
    initial = [_public_record(record) for record in initial_records]
    extended = [_public_record(record) for record in extended_records]
    total = initial + extended

    manifest_notes = list(notes or [])
    if len(total) < initial_size:
        manifest_notes.append(
            f"Only {len(total)} valid Task 1 records were available; "
            "the initial set contains all records and the extended set is empty."
        )
    else:
        manifest_notes.append(
            "The initial set reached the requested size without fabricating records."
        )
    manifest_notes.append(
        "All valid records included city, difficulty, and correct-answer category, "
        "so the full stratification fields were available."
    )
    missing_difficulties = sorted(
        VALID_DIFFICULTIES - {str(record["difficulty"]) for record in total}
    )
    if missing_difficulties:
        manifest_notes.append(
            "No valid records were available for these difficulty levels: "
            + ", ".join(missing_difficulties)
            + "."
        )

    return {
        "task": TASK1_NAME,
        "source_paths": [str(Path(path)) for path in source_paths],
        "total_valid_records": len(total),
        "initial_size_requested": initial_size,
        "initial_records": len(initial),
        "extended_records": len(extended),
        "random_seed": seed,
        "split_strategy": (
            "Deterministic stratified selection across city, difficulty, and "
            "correct-answer POI category where possible, with seeded tie-breaking."
        ),
        "created_at": created_at
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duplicate_records_removed": duplicate_records_removed,
        "invalid_records_excluded": invalid_records_excluded,
        "city_counts_total": _counts(total, "city"),
        "city_counts_initial": _counts(initial, "city"),
        "city_counts_extended": _counts(extended, "city"),
        "difficulty_counts_total": _counts(total, "difficulty"),
        "difficulty_counts_initial": _counts(initial, "difficulty"),
        "difficulty_counts_extended": _counts(extended, "difficulty"),
        "answer_category_counts_total": _counts(total, "correct_answer"),
        "answer_category_counts_initial": _counts(initial, "correct_answer"),
        "answer_category_counts_extended": _counts(extended, "correct_answer"),
        "notes": manifest_notes,
    }
