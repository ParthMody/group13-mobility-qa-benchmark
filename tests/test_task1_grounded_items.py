import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ITEMS_PATH = ROOT / "data/task1_items.jsonl"
CITY_CHECKIN_PATHS = {
    "Jakarta": ROOT / "data/jakarta_checkins.csv",
    "Sydney": ROOT / "data/sydney_checkins.csv",
    "Melbourne": ROOT / "data/melbourne_checkins.csv",
    "Tokyo": ROOT / "data/tokyo_checkins.csv",
    "New York": ROOT / "data/new-york_checkins.csv",
}


def load_items():
    return [
        json.loads(line)
        for line in ITEMS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_task1_uses_five_verified_massive_steps_items():
    items = load_items()

    assert len(items) == 5
    assert len({item["question_id"] for item in items}) == 5
    assert len({item["city"] for item in items}) == 5
    assert len({item["answer"] for item in items}) == 5
    assert {item["metadata"]["difficulty"] for item in items} == {
        "easy",
        "medium",
        "hard",
    }

    for item in items:
        assert item["task"] == "task1_next_poi_category"
        assert item["source_dataset"] == "massive_steps"
        assert item["user_id"] == "ANONYMISED"
        assert item["context_sequence"]
        assert len(item["choices"]) == 4
        assert item["answer"] in item["choices"]

        metadata = item["metadata"]
        assert metadata["answer_type"] == "closed"
        assert metadata["eval_mode"] == "classification"
        assert metadata["source_split"] == "test"
        assert metadata["verification_status"] == (
            "verified_massive_steps_taxonomy_mapping"
        )
        assert metadata["source_trail_id"] in item["question_id"]
        assert len(metadata["target_category_id"]) == 24
        assert metadata["target_fine_category"] in item["rationale"]


def test_task1_context_does_not_expose_user_or_venue_ids():
    for item in load_items():
        assert all("poi_id" not in checkin for checkin in item["context_sequence"])
        assert "draft_" not in json.dumps(item)


def test_task1_context_categories_match_the_source_trails():
    for item in load_items():
        source_path = CITY_CHECKIN_PATHS[item["city"]]
        source_trail_id = item["metadata"]["source_trail_id"]
        with source_path.open(encoding="utf-8", newline="") as source_file:
            source_categories = [
                row["venue_category"]
                for row in csv.DictReader(source_file)
                if row["trail_id"] == source_trail_id
            ]

        context_categories = [
            checkin["poi_category"] for checkin in item["context_sequence"]
        ]
        assert context_categories == source_categories
