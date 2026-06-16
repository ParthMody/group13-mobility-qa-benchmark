from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mobility_qa.io import read_checkins_csv, read_jsonl, write_jsonl  # noqa: E402


def test_jsonl_read_write_works(tmp_path):
    path = tmp_path / "records.jsonl"
    records = [
        {"question_id": "task1_001", "answer": "restaurant"},
        {"question_id": "task1_002", "answer": "library"},
    ]

    write_jsonl(records, path)

    assert read_jsonl(path) == records


def test_read_checkins_csv_validates_columns(tmp_path):
    path = tmp_path / "checkins.csv"
    path.write_text(
        "\n".join(
            [
                "user_id,venue_id,venue_category,timestamp,venue_city,latitude,longitude",
                "user_001,venue_001,home,2024-01-01T08:00:00,Sydney,-33.8688,151.2093",
            ]
        ),
        encoding="utf-8",
    )

    df = read_checkins_csv(path)

    assert list(df.columns) == [
        "user_id",
        "venue_id",
        "venue_category",
        "timestamp",
        "venue_city",
        "latitude",
        "longitude",
    ]
