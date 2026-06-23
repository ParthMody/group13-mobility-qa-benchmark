"""Schema checks for mobility check-ins and shared QA records."""

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

ALLOWED_SOURCE_DATASETS = {"massive_steps"}
ALLOWED_ANSWER_TYPES = {"closed", "open"}
ALLOWED_DIFFICULTIES = {"easy", "medium", "hard"}
TASK1_NAME = "task1_next_poi_category"


def _missing_fields(actual_fields: Iterable[str], required_fields: Iterable[str]) -> list[str]:
    actual = set(actual_fields)
    return [field for field in required_fields if field not in actual]


def validate_checkin_dataframe(df: pd.DataFrame) -> bool:
    """Validate the raw check-in CSV shape.

    Returns True for valid input and raises ValueError with a clear message for
    invalid input.
    """
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
    """Validate one generated QA record.

    Returns True for valid input and raises ValueError with a clear message for
    invalid input.
    """
    if not isinstance(record, Mapping):
        raise ValueError("QA record must be a mapping/dictionary.")

    missing = _missing_fields(record.keys(), REQUIRED_QA_FIELDS)
    if missing:
        raise ValueError(f"Missing required QA fields: {', '.join(missing)}")

    for field in [
        "question_id",
        "task",
        "city",
        "user_id",
        "target_time",
        "question",
        "rationale",
        "source_dataset",
    ]:
        value = record[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"QA field '{field}' must be a non-empty string.")

    if record["source_dataset"] not in ALLOWED_SOURCE_DATASETS:
        raise ValueError("QA field 'source_dataset' must be 'massive_steps'.")

    if not isinstance(record["context_sequence"], list) or not record["context_sequence"]:
        raise ValueError("QA field 'context_sequence' must be a non-empty list.")

    choices = record["choices"]
    if not isinstance(choices, list):
        raise ValueError("QA field 'choices' must be a list.")

    if not all(isinstance(choice, str) and choice.strip() for choice in choices):
        raise ValueError("QA field 'choices' must contain non-empty strings.")

    answer = record["answer"]
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("QA field 'answer' must be a non-empty string.")

    metadata = record["metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError("QA field 'metadata' must be a dictionary.")

    answer_type = metadata.get("answer_type")
    if answer_type not in ALLOWED_ANSWER_TYPES:
        raise ValueError("QA metadata 'answer_type' must be either 'closed' or 'open'.")

    if "eval_mode" not in metadata or not str(metadata["eval_mode"]).strip():
        raise ValueError("QA metadata 'eval_mode' must exist and be non-empty.")

    difficulty = metadata.get("difficulty")
    if difficulty not in ALLOWED_DIFFICULTIES:
        raise ValueError(
            "QA metadata 'difficulty' must be one of: easy, medium, hard."
        )

    if answer_type == "closed":
        if not choices:
            raise ValueError("Closed QA records must have a non-empty choices list.")
        if answer not in choices:
            raise ValueError("Closed QA field 'answer' must appear in 'choices'.")

    if answer_type == "open" and not answer.strip():
        raise ValueError("Open QA records must have a non-empty written answer.")

    return True


def validate_qa_records(records: Iterable[Mapping[str, object]]) -> bool:
    """Validate a collection of generated QA records."""
    for index, record in enumerate(records):
        try:
            validate_qa_record(record)
        except ValueError as exc:
            raise ValueError(f"Invalid QA record at index {index}: {exc}") from exc
    return True


def validate_task1_record(record: Mapping[str, object]) -> bool:
    """Validate one Task 1 Next POI Category QA record."""
    validate_qa_record(record)

    if record["task"] != TASK1_NAME:
        raise ValueError(f"Task 1 record must use task '{TASK1_NAME}'.")

    metadata = record["metadata"]
    if metadata.get("answer_type") != "closed":
        raise ValueError("Task 1 records must use metadata.answer_type = 'closed'.")

    if metadata.get("eval_mode") != "classification":
        raise ValueError("Task 1 records must use metadata.eval_mode = 'classification'.")

    return True
