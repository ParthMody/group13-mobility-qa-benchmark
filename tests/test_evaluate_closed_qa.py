from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mobility_qa.evaluation.evaluate_closed_qa import evaluate_accuracy  # noqa: E402


def closed_record(question_id: str, correct_answer: str) -> dict:
    return {
        "question_id": question_id,
        "answer_type": "multiple choice",
        "correct_answer": correct_answer,
    }


def test_closed_qa_accuracy_works():
    gold_records = [
        closed_record("q1", "Office"),
        closed_record("q2", "Train Station"),
        closed_record("q3", "Residential Area"),
    ]
    predictions = [
        {"question_id": "q1", "prediction": " OFFICE "},
        {"question_id": "q2", "prediction": "Park"},
    ]

    result = evaluate_accuracy(gold_records, predictions)
    assert result["count"] == 3
    assert result["correct"] == 1
    assert result["missing"] == 1
    assert result["accuracy"] == 1 / 3
