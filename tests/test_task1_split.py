from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mobility_qa.task1_split import (  # noqa: E402
    TASK1_NAME,
    create_task1_initial_extended_split,
    create_task1_split_manifest,
    deduplicate_task1_records,
    load_task1_records,
    validate_task1_records,
    write_task1_split,
)


def make_record(
    index: int,
    *,
    task: str = TASK1_NAME,
    city: str | None = None,
    difficulty: str | None = None,
    answer: str | None = None,
) -> dict:
    selected_city = city or ["Sydney", "Tokyo", "Jakarta"][index % 3]
    selected_difficulty = difficulty or ["easy", "medium", "hard"][index % 3]
    selected_answer = answer or ["Food", "Residence", "Travel"][index % 3]
    return {
        "question_id": f"task1_test_{index:04d}",
        "task": task,
        "city": selected_city,
        "context": f"POI sequence for synthetic user {index}",
        "answer_type": "multiple choice",
        "options": [selected_answer, "Alternative"],
        "correct_answer": selected_answer,
        "reasoning": "The synthetic held-out target maps to the selected category.",
        "difficulty": selected_difficulty,
        "source_dataset": "massive_steps",
        "verification_status": "synthetic_test",
    }


def test_records_from_other_tasks_are_excluded():
    valid, rejected = validate_task1_records(
        [make_record(1), make_record(2, task="task2_weekday_weekend")]
    )

    assert [record["question_id"] for record in valid] == ["task1_test_0001"]
    assert len(rejected) == 1
    assert "task must equal" in rejected[0]["rejection_reason"]


def test_fewer_than_400_places_every_record_in_initial():
    records = [make_record(index) for index in range(12)]

    initial, extended = create_task1_initial_extended_split(records)

    assert len(initial) == 12
    assert extended == []


def test_exactly_400_produces_no_extended_records():
    records = [make_record(index) for index in range(400)]

    initial, extended = create_task1_initial_extended_split(records)

    assert len(initial) == 400
    assert extended == []


def test_more_than_400_produces_exactly_400_initial_records():
    records = [make_record(index) for index in range(417)]

    initial, extended = create_task1_initial_extended_split(records)

    assert len(initial) == 400


def test_remaining_records_are_placed_in_extended():
    records = [make_record(index) for index in range(417)]

    _, extended = create_task1_initial_extended_split(records)

    assert len(extended) == 17


def test_no_question_id_appears_in_both_splits():
    records = [make_record(index) for index in range(405)]
    initial, extended = create_task1_initial_extended_split(records)

    initial_ids = {record["question_id"] for record in initial}
    extended_ids = {record["question_id"] for record in extended}

    assert initial_ids.isdisjoint(extended_ids)
    assert initial_ids | extended_ids == {
        record["question_id"] for record in records
    }


def test_identical_duplicates_are_removed_and_conflicts_fail():
    record = make_record(1)
    unique, duplicates = deduplicate_task1_records([record, dict(record)])

    assert len(unique) == 1
    assert len(duplicates) == 1

    conflict = dict(record)
    conflict["city"] = "Melbourne"
    with pytest.raises(ValueError, match="Conflicting duplicate question_id"):
        deduplicate_task1_records([record, conflict])


def test_invalid_multiple_choice_records_are_excluded():
    empty_options = make_record(1)
    empty_options["options"] = []
    wrong_type = make_record(2)
    wrong_type["answer_type"] = "written"

    valid, rejected = validate_task1_records([empty_options, wrong_type])

    assert valid == []
    assert len(rejected) == 2


def test_correct_answer_must_appear_in_options():
    record = make_record(1)
    record["correct_answer"] = "Park"

    valid, rejected = validate_task1_records([record])

    assert valid == []
    assert "correct_answer must appear exactly in options" in rejected[0][
        "rejection_reason"
    ]


def test_split_is_reproducible_with_seed_5925():
    records = [make_record(index) for index in range(430)]
    first = create_task1_initial_extended_split(records, seed=5925)
    second = create_task1_initial_extended_split(list(reversed(records)), seed=5925)

    assert first == second


def representative_records() -> list[dict]:
    return [
        make_record(
            index,
            city=["Sydney", "Tokyo", "Jakarta", "Melbourne"][index % 4],
            difficulty=["easy", "medium", "hard"][index % 3],
            answer=["Food", "Residence", "Travel", "Outdoors"][index % 4],
        )
        for index in range(60)
    ]


def test_different_cities_and_answers_are_represented_where_possible():
    initial, _ = create_task1_initial_extended_split(
        representative_records(), initial_size=12, seed=5925
    )

    assert {record["city"] for record in initial} == {
        "Sydney",
        "Tokyo",
        "Jakarta",
        "Melbourne",
    }
    assert {record["correct_answer"] for record in initial} == {
        "Food",
        "Residence",
        "Travel",
        "Outdoors",
    }


def test_difficulty_levels_are_represented_where_possible():
    records = [
        make_record(
            index,
            city=["Sydney", "Tokyo", "Jakarta", "Melbourne"][index % 4],
            difficulty=["easy", "medium", "hard"][index % 3],
            answer=["Food", "Residence", "Travel", "Outdoors"][index % 4],
        )
        for index in range(60)
    ]
    initial, _ = create_task1_initial_extended_split(
        records, initial_size=12, seed=5925
    )

    assert {record["difficulty"] for record in initial} == {
        "easy",
        "medium",
        "hard",
    }


def test_manifest_counts_match_split_outputs():
    records = [make_record(index) for index in range(405)]
    initial, extended = create_task1_initial_extended_split(records)

    manifest = create_task1_split_manifest(
        initial,
        extended,
        source_paths=["synthetic.jsonl"],
        duplicate_records_removed=2,
        invalid_records_excluded=3,
        created_at="2026-07-24T00:00:00+00:00",
    )

    assert manifest["total_valid_records"] == 405
    assert manifest["initial_records"] == 400
    assert manifest["extended_records"] == 5
    assert manifest["duplicate_records_removed"] == 2
    assert manifest["invalid_records_excluded"] == 3
    assert sum(manifest["city_counts_total"].values()) == 405
    assert sum(manifest["difficulty_counts_initial"].values()) == 400
    assert sum(manifest["answer_category_counts_extended"].values()) == 5


def test_csv_and_jsonl_outputs_are_equivalent(tmp_path):
    records = [make_record(index) for index in range(8)]
    csv_path = tmp_path / "task1.csv"
    jsonl_path = tmp_path / "task1.jsonl"

    write_task1_split(records, csv_path, jsonl_path)

    csv_records = load_task1_records(csv_path)
    jsonl_records = load_task1_records(jsonl_path)
    assert csv_records == jsonl_records
    assert len(json.loads(jsonl_path.read_text().splitlines()[0])["options"]) == 2
