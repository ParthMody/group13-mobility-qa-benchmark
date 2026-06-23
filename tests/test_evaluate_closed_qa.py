from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mobility_qa.evaluation.evaluate_closed_qa import evaluate_accuracy  # noqa: E402


def closed_record(question_id: str, answer: str) -> dict:
    return {
        "question_id": question_id,
        "answer": answer,
        "metadata": {"answer_type": "closed"},
    }


def test_closed_qa_accuracy_works():
    gold_records = [
        closed_record("q1", "Office"),
        closed_record("q2", "Restaurant"),
        closed_record("q3", "Residence"),
    ]
    pred_records = [
        {"question_id": "q1", "prediction": " office "},
        {"question_id": "q2", "prediction": "Cafe"},
    ]

    result = evaluate_accuracy(gold_records, pred_records)

    assert result["count"] == 3
    assert result["correct"] == 1
    assert result["missing"] == 1
    assert result["accuracy"] == 1 / 3
