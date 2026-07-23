"""Schema checks for mobility check-ins and QA records."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd


REQUIRED_CHECKIN_COLUMNS = [
    "user_id",
    "venue_id",
    "venue_category",
    "timestamp",
    "venue_city",
    "latitude",
    "longitude",
]

REQUIRED_QA_FIELDS = [
    "question_id",
    "task",
    "city",
    "user_id",
    "context_sequence",
    "target_time",
    "question",
    "choices",
    "answer",
    "rationale",
    "source_dataset",
    "metadata",
]


def _missing_fields(
    actual_fields: Iterable[str], required_fields: Iterable[str]
) -> list[str]:
    actual = set(actual_fields)
    return [field for field in required_fields if field not in actual]


def validate_checkin_dataframe(df: pd.DataFrame) -> bool:
    """Validate the raw check-in CSV shape."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Check-in data must be a pandas DataFrame.")

    missing = _missing_fields(df.columns, REQUIRED_CHECKIN_COLUMNS)
    if missing:
        raise ValueError(f"Missing required check-in columns: {', '.join(missing)}")

    null_columns = [
        column for column in REQUIRED_CHECKIN_COLUMNS if df[column].isna().any()
    ]
    if null_columns:
        raise ValueError(
            "Check-in columns contain missing values: " + ", ".join(null_columns)
        )

    for column in ["latitude", "longitude"]:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"Check-in column must be numeric: {column}")

    return True


def validate_qa_record(record: Mapping[str, object]) -> bool:
    """Validate one generated QA record."""
    if not isinstance(record, Mapping):
        raise ValueError("QA record must be a mapping/dictionary.")

    missing = _missing_fields(record.keys(), REQUIRED_QA_FIELDS)
    if missing:
        raise ValueError(f"Missing required QA fields: {', '.join(missing)}")

    context_sequence = record["context_sequence"]
    if not isinstance(context_sequence, list) or not context_sequence:
        raise ValueError("QA field 'context_sequence' must be a non-empty list.")

    choices = record["choices"]
    if not isinstance(choices, list) or len(choices) < 2:
        raise ValueError("QA field 'choices' must be a list with at least 2 items.")

    if not all(isinstance(choice, str) and choice for choice in choices):
        raise ValueError("QA field 'choices' must contain non-empty strings.")

    answer = record["answer"]
    if not isinstance(answer, str) or not answer:
        raise ValueError("QA field 'answer' must be a non-empty string.")

    if answer not in choices:
        raise ValueError("QA field 'answer' must appear in 'choices'.")

    if not isinstance(record["metadata"], Mapping):
        raise ValueError("QA field 'metadata' must be a dictionary.")

    return True


def validate_qa_records(records: Iterable[Mapping[str, object]]) -> bool:
    """Validate a collection of generated QA records."""
    for index, record in enumerate(records):
        try:
            validate_qa_record(record)
        except ValueError as exc:
            raise ValueError(f"Invalid QA record at index {index}: {exc}") from exc
    return True
