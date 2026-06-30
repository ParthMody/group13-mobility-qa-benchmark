"""Input and output helpers for the mobility QA benchmark."""

from __future__ import annotations

import json
import csv
from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd

from mobility_qa.schema import validate_checkin_dataframe


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file into a list of dictionaries."""
    records = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    return records


def write_jsonl(records: Iterable[Mapping[str, object]], path: str | Path) -> None:
    """Write records to a JSONL file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def _encode_csv_value(field: str, value: object) -> object:
    if field in {"options", "choices"}:
        if not isinstance(value, list):
            raise ValueError(f"CSV field '{field}' must be a list before writing.")
        return " | ".join(str(option) for option in value)

    if field == "metadata":
        if not isinstance(value, Mapping):
            raise ValueError("CSV field 'metadata' must be a dictionary before writing.")
        return json.dumps(value, ensure_ascii=True, sort_keys=True)

    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)

    return value


def _decode_context_sequence(value: str) -> object:
    if not value:
        return []
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return value
    return decoded


def read_csv_records(path: str | Path) -> list[dict]:
    """Read shared QA records from CSV.

    Option lists are pipe-separated. Legacy shared-format fields remain
    readable so older example files do not break the helper.
    """
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            record = dict(row)
            for field in ("options", "choices"):
                if field in record:
                    option_text = record[field]
                    record[field] = (
                        [option.strip() for option in option_text.split("|")]
                        if option_text
                        else []
                    )
            if "metadata" in record:
                record["metadata"] = json.loads(record["metadata"] or "{}")
            if "context_sequence" in record:
                record["context_sequence"] = _decode_context_sequence(
                    record["context_sequence"]
                )
            records.append(record)
    return records


def write_csv_records(records: Iterable[Mapping[str, object]], path: str | Path) -> None:
    """Write shared QA records to CSV with stable JSON/choice encoding."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records_list = list(records)
    if not records_list:
        with output_path.open("w", encoding="utf-8", newline="") as file:
            file.write("")
        return

    fieldnames = list(records_list[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for record in records_list:
            writer.writerow(
                {
                    field: _encode_csv_value(field, record.get(field, ""))
                    for field in fieldnames
                }
            )


def read_checkins_csv(path: str | Path) -> pd.DataFrame:
    """Read and validate a raw check-in CSV file."""
    df = pd.read_csv(path)
    validate_checkin_dataframe(df)
    return df
