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
                "venue_category": "Residence",
                "timestamp": "2024-01-01T08:00:00",
                "venue_city": "Tokyo",
                "latitude": 35.6762,
                "longitude": 139.6503,
            }
        ]
    )


def valid_closed_task1_record() -> dict:
    return {
        "question_id": "task1_001",
        "task": "task1_next_poi_category",
        "city": "Tokyo",
        "user_id": "user_001",
        "context_sequence": [
            {
                "poi_id": "poi_001",
                "poi_category": "Residence",
                "timestamp": "2024-01-01T08:00:00",
            }
        ],
        "target_time": "2024-01-01T09:00:00",
        "question": "What broad POI category is the most likely next stop?",
        "choices": ["Office", "Cafe", "Park", "Restaurant"],
        "answer": "Office",
        "rationale": "Morning commute pattern.",
        "source_dataset": "massive_steps",
        "metadata": {
            "answer_type": "closed",
            "eval_mode": "classification",
            "difficulty": "easy",
        },
    }


def test_valid_checkin_dataframe_passes():
    assert validate_checkin_dataframe(valid_checkin_dataframe()) is True


def test_missing_checkin_columns_fail():
    df = valid_checkin_dataframe().drop(columns=["venue_category"])

    with pytest.raises(ValueError, match="Missing required check-in columns"):
        validate_checkin_dataframe(df)


def test_valid_closed_task1_record_passes():
    assert validate_task1_record(valid_closed_task1_record()) is True


def test_closed_record_fails_if_answer_not_in_choices():
    record = valid_closed_task1_record()
    record["answer"] = "Hotel"

    with pytest.raises(ValueError, match="answer.*choices"):
        validate_qa_record(record)


def test_open_record_can_have_empty_choices():
    record = valid_closed_task1_record()
    record["task"] = "task4_open_reasoning"
    record["choices"] = []
    record["answer"] = "The user likely returns home after transit."
    record["metadata"] = {
        "answer_type": "open",
        "eval_mode": "rubric",
        "difficulty": "medium",
    }

    assert validate_qa_record(record) is True


def test_missing_answer_type_fails():
    record = valid_closed_task1_record()
    record["metadata"].pop("answer_type")

    with pytest.raises(ValueError, match="answer_type"):
        validate_qa_record(record)
