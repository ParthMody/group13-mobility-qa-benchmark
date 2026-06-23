"""Task 1 utilities for Next POI Category QA."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

TASK_NAME = "task1_next_poi_category"

from mobility_qa.io import write_csv_records, write_jsonl
from mobility_qa.schema import validate_task1_record


QUESTION_TEXT = (
    "Given the user's recent check-ins, what broad POI category is the most "
    "likely next stop?"
)

DEFAULT_METADATA = {
    "answer_type": "closed",
    "eval_mode": "classification",
    "verification_status": "draft_category_example_pending_raw_mapping",
}


def build_task1_question(record: Mapping[str, Any]) -> str:
    """Build the natural-language Task 1 question for one context record."""
    city = record.get("city", "the city")
    target_time = record.get("target_time", "the target time")
    return (
        f"Given the user's recent check-ins in {city}, what broad POI category "
        f"is the most likely next stop at {target_time}?"
    )


def create_task1_record(
    question_id: str,
    city: str,
    user_id: str,
    context_sequence: list[dict[str, Any]],
    target_time: str,
    choices: list[str],
    answer: str,
    rationale: str,
    difficulty: str = "easy",
) -> dict[str, Any]:
    """Create and validate one shared-format Task 1 record."""
    record: dict[str, Any] = {
        "question_id": question_id,
        "task": TASK_NAME,
        "city": city,
        "user_id": user_id,
        "context_sequence": context_sequence,
        "target_time": target_time,
        "question": QUESTION_TEXT,
        "choices": choices,
        "answer": answer,
        "rationale": rationale,
        "source_dataset": "massive_steps",
        "metadata": {
            **DEFAULT_METADATA,
            "difficulty": difficulty,
            "history_length": len(context_sequence),
        },
    }
    record["question"] = build_task1_question(record)
    validate_task1_record(record)
    return record


def _draft_examples() -> list[dict[str, Any]]:
    return [
        create_task1_record(
            question_id="task1_dhanesh_001",
            city="Tokyo",
            user_id="ms_user_0001",
            context_sequence=[
                {
                    "poi_id": "draft_tokyo_home_001",
                    "poi_category": "Residence",
                    "timestamp": "2024-03-04T07:35:00",
                },
                {
                    "poi_id": "draft_tokyo_transit_014",
                    "poi_category": "Travel & Transport",
                    "timestamp": "2024-03-04T08:05:00",
                },
                {
                    "poi_id": "draft_tokyo_cafe_022",
                    "poi_category": "Cafe",
                    "timestamp": "2024-03-04T08:30:00",
                },
            ],
            target_time="2024-03-04T08:55:00",
            choices=["Office", "Cafe", "Park", "Restaurant"],
            answer="Office",
            rationale=(
                "The weekday sequence follows a commute from residence through "
                "transport with a short cafe stop, so the user is likely returning "
                "to the office."
            ),
            difficulty="easy",
        ),
        create_task1_record(
            question_id="task1_dhanesh_002",
            city="New York",
            user_id="ms_user_0002",
            context_sequence=[
                {
                    "poi_id": "draft_nyc_office_011",
                    "poi_category": "Office",
                    "timestamp": "2024-03-05T09:00:00",
                },
                {
                    "poi_id": "draft_nyc_cafe_026",
                    "poi_category": "Cafe",
                    "timestamp": "2024-03-05T10:35:00",
                },
                {
                    "poi_id": "draft_nyc_office_011",
                    "poi_category": "Office",
                    "timestamp": "2024-03-05T11:05:00",
                },
            ],
            target_time="2024-03-05T12:25:00",
            choices=["Gym / Fitness Center", "Restaurant", "Residence", "Museum"],
            answer="Restaurant",
            rationale=(
                "After a morning office and cafe pattern, the next midday stop "
                "is most plausibly a lunch venue."
            ),
            difficulty="easy",
        ),
        create_task1_record(
            question_id="task1_dhanesh_003",
            city="Singapore",
            user_id="ms_user_0003",
            context_sequence=[
                {
                    "poi_id": "draft_sg_residence_004",
                    "poi_category": "Residence",
                    "timestamp": "2024-03-09T10:10:00",
                },
                {
                    "poi_id": "draft_sg_park_032",
                    "poi_category": "Outdoors & Recreation",
                    "timestamp": "2024-03-09T11:00:00",
                },
                {
                    "poi_id": "draft_sg_museum_009",
                    "poi_category": "Arts & Entertainment",
                    "timestamp": "2024-03-09T13:20:00",
                },
            ],
            target_time="2024-03-09T15:05:00",
            choices=["Office", "Arts & Entertainment", "Medical Center", "School"],
            answer="Arts & Entertainment",
            rationale=(
                "The weekend sequence already shows leisure movement from a "
                "park to an entertainment venue, so another leisure category is likely."
            ),
            difficulty="medium",
        ),
        create_task1_record(
            question_id="task1_dhanesh_004",
            city="San Francisco",
            user_id="ms_user_0004",
            context_sequence=[
                {
                    "poi_id": "draft_sf_office_042",
                    "poi_category": "Office",
                    "timestamp": "2024-03-06T17:20:00",
                },
                {
                    "poi_id": "draft_sf_gym_007",
                    "poi_category": "Gym / Fitness Center",
                    "timestamp": "2024-03-06T18:15:00",
                },
                {
                    "poi_id": "draft_sf_transit_018",
                    "poi_category": "Travel & Transport",
                    "timestamp": "2024-03-06T19:25:00",
                },
            ],
            target_time="2024-03-06T20:00:00",
            choices=["Residence", "Restaurant", "Office", "Nightlife Spot"],
            answer="Residence",
            rationale=(
                "The evening pattern moves from work to gym to transport, which "
                "commonly ends at residence."
            ),
            difficulty="easy",
        ),
        create_task1_record(
            question_id="task1_dhanesh_005",
            city="Los Angeles",
            user_id="ms_user_0005",
            context_sequence=[
                {
                    "poi_id": "draft_la_residence_016",
                    "poi_category": "Residence",
                    "timestamp": "2024-03-07T15:40:00",
                },
                {
                    "poi_id": "draft_la_shop_023",
                    "poi_category": "Shop & Service",
                    "timestamp": "2024-03-07T16:30:00",
                },
                {
                    "poi_id": "draft_la_transit_031",
                    "poi_category": "Travel & Transport",
                    "timestamp": "2024-03-07T17:15:00",
                },
            ],
            target_time="2024-03-07T17:55:00",
            choices=["Shop & Service", "College & University", "Residence", "Park"],
            answer="Residence",
            rationale=(
                "The user leaves a shopping stop and then uses transport, making "
                "a return to residence the most likely next broad category."
            ),
            difficulty="medium",
        ),
    ]


def export_task1_examples(output_csv: str | Path, output_jsonl: str | Path) -> list[dict[str, Any]]:
    """Export the five Dhanesh Task 1 draft examples to CSV and JSONL."""
    records = _draft_examples()
    write_csv_records(records, output_csv)
    write_jsonl(records, output_jsonl)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Task 1 draft examples.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-jsonl", required=True)
    args = parser.parse_args()

    export_task1_examples(args.output_csv, args.output_jsonl)


if __name__ == "__main__":
    main()
