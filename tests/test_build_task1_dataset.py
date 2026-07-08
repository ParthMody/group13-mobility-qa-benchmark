from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mobility_qa.schema import validate_task1_record  # noqa: E402
from mobility_qa.tasks.build_task1_dataset import (  # noqa: E402
    build_task1_records,
    flatten_foursquare_taxonomy,
    massive_steps_url,
)


def taxonomy() -> list[dict]:
    return [
        {
            "id": "food-root",
            "name": "Food",
            "categories": [
                {"id": "cafe", "name": "Cafe", "categories": []},
                {"id": "restaurant", "name": "Restaurant", "categories": []},
            ],
        },
        {
            "id": "travel-root",
            "name": "Travel & Transport",
            "categories": [
                {"id": "station", "name": "Train Station", "categories": []}
            ],
        },
        {
            "id": "work-root",
            "name": "Professional & Other Places",
            "categories": [
                {"id": "office", "name": "Office", "categories": []}
            ],
        },
        {
            "id": "shop-root",
            "name": "Shop & Service",
            "categories": [
                {"id": "mall", "name": "Shopping Mall", "categories": []}
            ],
        },
    ]


def checkins() -> pd.DataFrame:
    rows = []
    trails = [
        ("trail_1", [("station", "Train Station"), ("office", "Office"), ("cafe", "Cafe")]),
        ("trail_2", [("office", "Office"), ("cafe", "Cafe"), ("cafe", "Cafe")]),
        ("trail_3", [("mall", "Shopping Mall"), ("cafe", "Cafe"), ("restaurant", "Restaurant")]),
        ("trail_4", [("station", "Train Station"), ("mall", "Shopping Mall"), ("station", "Train Station")]),
    ]
    for trail_id, visits in trails:
        for index, (category_id, category) in enumerate(visits):
            rows.append(
                {
                    "trail_id": trail_id,
                    "venue_category": category,
                    "venue_category_id": category_id,
                    "timestamp": f"2018-01-01 {index + 8:02d}:00:00",
                    "venue_id": 100 + index,
                    "name": f"Venue {index}",
                }
            )
    return pd.DataFrame(rows)


def test_flatten_taxonomy_maps_leaf_ids_to_top_level():
    mapping = flatten_foursquare_taxonomy(taxonomy())

    assert mapping["cafe"] == "Food"
    assert mapping["station"] == "Travel & Transport"
    assert mapping["office"] == "Professional & Other Places"


def test_task1_builder_is_deterministic_and_hides_venue_details():
    mapping = flatten_foursquare_taxonomy(taxonomy())

    records, stats = build_task1_records(
        checkins(), "Bandung", "bandung", "test", mapping, limit=4
    )
    repeated, _ = build_task1_records(
        checkins(), "Bandung", "bandung", "test", mapping, limit=4
    )

    assert records == repeated
    assert stats["selected_records"] == 4
    assert {record["difficulty"] for record in records} == {"easy", "medium", "hard"}
    for record in records:
        assert validate_task1_record(record) is True
        assert len(record["options"]) == 4
        assert record["correct_answer"] in record["options"]
        assert "Venue " not in record["context"]
        assert "POI id" not in record["context"]


def test_unknown_taxonomy_ids_are_skipped():
    frame = checkins()
    frame.loc[frame["trail_id"] == "trail_1", "venue_category_id"] = "unknown"
    mapping = flatten_foursquare_taxonomy(taxonomy())

    records, stats = build_task1_records(
        frame, "Bandung", "bandung", "test", mapping, limit=10
    )

    assert len(records) == 3
    assert stats["skipped_unmapped"] == 1


def test_official_download_url_uses_city_and_split():
    url = massive_steps_url("new_york", "test")

    assert "Massive-STEPS-New-York" in url
    assert "new_york_checkins_test.csv" in url
