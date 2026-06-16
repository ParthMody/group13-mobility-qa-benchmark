from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mobility_qa.schema import (  # noqa: E402
    validate_checkin_dataframe,
    validate_qa_record,
)


def valid_checkin_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "user_id": "user_001",
                "venue_id": "venue_001",
                "venue_category": "home",
                "timestamp": "2024-01-01T08:00:00",
                "venue_city": "Sydney",
                "latitude": -33.8688,
                "longitude": 151.2093,
            }
        ]
    )


def valid_qa_record() -> dict:
    return {
        "question_id": "task1_001",
        "task": "task1_next_poi_category",
        "city": "Sydney",
        "user_id": "user_001",
        "context_sequence": [
            {
                "venue_id": "venue_001",
                "venue_category": "home",
                "timestamp": "2024-01-01T08:00:00",
            }
        ],
        "target_time": "2024-01-01T12:20:00",
        "question": "What is the most likely next POI category?",
        "choices": ["restaurant", "cafe", "gym", "home"],
        "answer": "restaurant",
        "rationale": "Synthetic example target category.",
        "source_dataset": "synthetic_example",
        "metadata": {"history_length": 1},
    }


def test_valid_checkin_dataframe_passes():
    assert validate_checkin_dataframe(valid_checkin_dataframe()) is True


def test_missing_checkin_columns_fail():
    df = valid_checkin_dataframe().drop(columns=["venue_category"])

    with pytest.raises(ValueError, match="Missing required check-in columns"):
        validate_checkin_dataframe(df)


def test_valid_qa_record_passes():
    assert validate_qa_record(valid_qa_record()) is True


def test_missing_qa_field_fails():
    record = valid_qa_record()
    record.pop("answer")

    with pytest.raises(ValueError, match="Missing required QA fields"):
        validate_qa_record(record)
