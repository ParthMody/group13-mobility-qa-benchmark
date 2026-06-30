from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mobility_qa.schema import (  # noqa: E402
    validate_checkin_dataframe,
    validate_qa_record,
    validate_task1_record,
)


def valid_checkin_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "user_id": "user_001",
                "venue_id": "venue_001",
                "venue_category": "Residential Area",
                "timestamp": "2024-01-01T08:00:00",
                "venue_city": "Tokyo",
                "latitude": 35.6762,
                "longitude": 139.6503,
            }
        ]
    )


def valid_task1_record() -> dict:
    return {
        "question_id": "task1_001",
        "task": "task1_next_poi_category",
        "city": "Jakarta",
        "context": "The user visits a station and then an office.",
        "answer_type": "multiple choice",
        "options": ["Office", "Park"],
        "correct_answer": "Office",
        "reasoning": "This follows a weekday commute pattern.",
        "difficulty": "easy",
        "source_dataset": "massive_steps",
        "verification_status": "draft_manual_example",
    }


def test_valid_checkin_dataframe_passes():
    assert validate_checkin_dataframe(valid_checkin_dataframe()) is True


def test_missing_checkin_columns_fail():
    df = valid_checkin_dataframe().drop(columns=["venue_category"])

    with pytest.raises(ValueError, match="Missing required check-in columns"):
        validate_checkin_dataframe(df)


def test_valid_task1_record_passes():
    assert validate_task1_record(valid_task1_record()) is True


def test_multiple_choice_answer_must_be_an_option():
    record = valid_task1_record()
    record["correct_answer"] = "Hospital"

    with pytest.raises(ValueError, match="correct_answer.*options"):
        validate_qa_record(record)


def test_written_record_can_have_empty_options():
    record = valid_task1_record()
    record["task"] = "task4_open_reasoning"
    record["answer_type"] = "written"
    record["options"] = []
    record["correct_answer"] = "The user likely returns home after transit."

    assert validate_qa_record(record) is True


def test_missing_answer_type_fails():
    record = valid_task1_record()
    record.pop("answer_type")

    with pytest.raises(ValueError, match="answer_type"):
        validate_qa_record(record)
