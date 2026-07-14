import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ITEMS_PATH = ROOT / "data/task1_items.jsonl"


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
