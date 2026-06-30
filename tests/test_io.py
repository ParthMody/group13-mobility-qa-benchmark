from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mobility_qa.io import (  # noqa: E402
    read_checkins_csv,
    read_csv_records,
    read_jsonl,
    write_csv_records,
    write_jsonl,
)


def qa_record() -> dict:
    return {
        "question_id": "task1_001",
        "task": "task1_next_poi_category",
        "city": "Jakarta",
        "context": "The user visits a station and then an office.",
        "answer_type": "multiple choice",
        "options": ["Office", "Park", "Hospital"],
        "correct_answer": "Office",
        "reasoning": "This follows a weekday commute pattern.",
        "difficulty": "easy",
        "source_dataset": "massive_steps",
        "verification_status": "draft_manual_example",
    }


def test_jsonl_read_write_roundtrip_works(tmp_path):
    path = tmp_path / "records.jsonl"
    records = [qa_record()]
    write_jsonl(records, path)
    assert read_jsonl(path) == records


def test_csv_read_write_roundtrip_preserves_options(tmp_path):
    path = tmp_path / "records.csv"
    records = [qa_record()]
    write_csv_records(records, path)
    assert read_csv_records(path) == records


def test_read_checkins_csv_validates_columns(tmp_path):
    path = tmp_path / "checkins.csv"
    path.write_text(
        "\n".join(
            [
                "user_id,venue_id,venue_category,timestamp,venue_city,latitude,longitude",
                "user_001,venue_001,Residence,2024-01-01T08:00:00,Tokyo,35.6762,139.6503",
            ]
        ),
        encoding="utf-8",
    )

    df = read_checkins_csv(path)
    assert len(df) == 1
