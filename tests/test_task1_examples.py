from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mobility_qa.evaluation.evaluate_closed_qa import evaluate_closed_qa  # noqa: E402
from mobility_qa.io import read_csv_records, read_jsonl, write_jsonl  # noqa: E402
from mobility_qa.schema import ALLOWED_DIFFICULTIES, validate_task1_record  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data/examples/task1_dhanesh_5_questions.csv"
JSONL_PATH = ROOT / "data/examples/task1_dhanesh_5_questions.jsonl"


def test_all_five_task1_examples_load_and_match():
    csv_records = read_csv_records(CSV_PATH)
    jsonl_records = read_jsonl(JSONL_PATH)

    assert len(csv_records) == 5
    assert csv_records == jsonl_records
    assert [record["question_id"] for record in jsonl_records] == [
        "task1_001",
        "task1_002",
        "task1_003",
        "task1_004",
        "task1_005",
    ]


def test_task1_examples_are_valid_multiple_choice_records():
    records = read_jsonl(JSONL_PATH)

    for record in records:
        assert validate_task1_record(record) is True
        assert record["options"]
        assert record["correct_answer"] in record["options"]
        assert record["difficulty"] in ALLOWED_DIFFICULTIES
        assert record["verification_status"] == "draft_manual_example"


def test_evaluation_accuracy_with_tiny_prediction_file(tmp_path):
    prediction_path = tmp_path / "predictions.jsonl"
    write_jsonl(
        [
            {"question_id": "task1_001", "prediction": " office  "},
            {"question_id": "task1_002", "prediction": "Park"},
            {"question_id": "task1_003", "prediction": " PARK"},
            {"question_id": "task1_004", "prediction": "residential area"},
            {"question_id": "task1_005", "prediction": "Hospital"},
        ],
        prediction_path,
    )

    result = evaluate_closed_qa(JSONL_PATH, prediction_path)
    assert result["count"] == 5
    assert result["correct"] == 3
    assert result["accuracy"] == 0.6
