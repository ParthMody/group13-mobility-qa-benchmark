"""Manual Task 1 examples for Next-POI Category QA."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from mobility_qa.io import write_csv_records, write_jsonl
from mobility_qa.schema import validate_task1_record


TASK_NAME = "task1_next_poi_category"
SOURCE_DATASET = "massive_steps"
VERIFICATION_STATUS = "draft_manual_example"


def build_task1_question(record: Mapping[str, Any]) -> str:
    """Return a category-level question using a record's context."""
    return (
        f"{str(record['context']).strip()} "
        "What is the most likely next broad POI category?"
    )


def create_task1_record(
    question_id: str,
    city: str,
    context: str,
    options: list[str],
    correct_answer: str,
    reasoning: str,
    difficulty: str = "easy",
) -> dict[str, Any]:
    """Create and validate one manual Task 1 record."""
    record = {
        "question_id": question_id,
        "task": TASK_NAME,
        "city": city,
        "context": context,
        "answer_type": "multiple choice",
        "options": options,
        "correct_answer": correct_answer,
        "reasoning": reasoning,
        "difficulty": difficulty,
        "source_dataset": SOURCE_DATASET,
        "verification_status": VERIFICATION_STATUS,
    }
    validate_task1_record(record)
    return record


def _manual_examples() -> list[dict[str, Any]]:
    return [
        create_task1_record(
            "task1_001",
            "Jakarta",
            "The user’s recent check-ins are Residential Area at 07:40, Train Station at 08:10, Office at 09:00, and Cafe at 12:30. The target time is 13:15 on the same day.",
            ["Office", "Park", "Nightlife", "Hospital"],
            "Office",
            "The sequence suggests a weekday work routine. The cafe visit is likely a lunch break, so the most likely next category is Office.",
            "easy",
        ),
        create_task1_record(
            "task1_002",
            "Bandung",
            "The user’s recent check-ins are Train Station at 08:05, Office at 08:50, Restaurant at 12:20, and Cafe at 13:00. The target time is 13:45 on the same day.",
            ["Office", "Mall", "Park", "Nightlife"],
            "Office",
            "The user appears to be following a workday pattern. After lunch and a short cafe stop, returning to the office is more plausible than going to leisure or nightlife locations.",
            "easy",
        ),
        create_task1_record(
            "task1_003",
            "Melbourne",
            "The user’s recent check-ins are Cafe at 10:15, Shopping Mall at 12:05, Restaurant at 13:30, and Cinema at 15:20. The target time is 17:00 on the same day.",
            ["Park", "Office", "Hospital", "University"],
            "Park",
            "The sequence shows a leisure-oriented day with cafe, shopping, restaurant, and cinema visits. Park is the most consistent next category among the options.",
            "medium",
        ),
        create_task1_record(
            "task1_004",
            "Sydney",
            "The user’s recent check-ins are Office at 09:10, Cafe at 11:30, Office at 13:00, and Gym at 18:20. The target time is 19:30 on the same day.",
            ["Residential Area", "Office", "Train Station", "Nightlife"],
            "Residential Area",
            "The pattern suggests a normal weekday routine. After work and an evening gym visit, the user is most likely to return to a residential area.",
            "medium",
        ),
        create_task1_record(
            "task1_005",
            "Tokyo",
            "The user’s recent check-ins are Residential Area at 07:30, Train Station at 08:00, Office at 08:45, Restaurant at 12:10, and Shopping Mall at 18:15. The target time is 19:10 on the same day.",
            ["Train Station", "Hospital", "University", "Office"],
            "Train Station",
            "The user starts the day with a commute to work and later visits a shopping mall in the evening. At 19:10, the most likely next category is Train Station as part of the return commute.",
            "hard",
        ),
    ]


def export_task1_examples(
    output_csv: str | Path, output_jsonl: str | Path
) -> list[dict[str, Any]]:
    """Export the five draft manual examples in CSV and JSONL formats."""
    records = _manual_examples()
    write_csv_records(records, output_csv)
    write_jsonl(records, output_jsonl)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Export manual Task 1 examples.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-jsonl", required=True)
    args = parser.parse_args()
    export_task1_examples(args.output_csv, args.output_jsonl)


if __name__ == "__main__":
    main()
