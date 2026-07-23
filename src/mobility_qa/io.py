"""Input and output helpers for the mobility QA benchmark."""

from __future__ import annotations

import json
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


def write_jsonl(
    records: Iterable[Mapping[str, object]], path: str | Path
) -> None:
    """Write records to a JSONL file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")


def read_checkins_csv(path: str | Path) -> pd.DataFrame:
    """Read and validate a raw check-in CSV file."""
    df = pd.read_csv(path)
    validate_checkin_dataframe(df)
    return df
