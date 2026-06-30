"""Exact-match evaluation for closed QA tasks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from mobility_qa.io import read_csv_records, read_jsonl


def normalize_answer(text: object) -> str:
    """Normalize a closed QA answer for exact-match comparison."""
    return " ".join(str(text).strip().lower().split())


def _is_closed_record(record: Mapping[str, Any]) -> bool:
    return record.get("answer_type") == "multiple choice"


def evaluate_accuracy(
    gold_records: list[Mapping[str, Any]],
    pred_records: list[Mapping[str, Any]],
    *,
    on_open: str = "skip",
) -> dict[str, float | int]:
    """Evaluate exact-match accuracy for closed QA records.

    Args:
        gold_records: Shared-format gold records with ``correct_answer``.
        pred_records: Prediction records with question_id and prediction.
        on_open: Use "skip" to ignore open records, or "error" to fail clearly.
    """
    if on_open not in {"skip", "error"}:
        raise ValueError("on_open must be either 'skip' or 'error'.")

    predictions = {
        str(record["question_id"]): record.get("prediction", "")
        for record in pred_records
        if "question_id" in record
    }

    total = 0
    correct = 0
    missing = 0
    skipped_open = 0

    for record in gold_records:
        if not _is_closed_record(record):
            if on_open == "error":
                question_id = record.get("question_id", "<missing question_id>")
                raise ValueError(f"Gold record is not closed: {question_id}")
            skipped_open += 1
            continue

        question_id = str(record["question_id"])
        total += 1

        if question_id not in predictions:
            missing += 1
            continue

        if normalize_answer(predictions[question_id]) == normalize_answer(
            record["correct_answer"]
        ):
            correct += 1

    accuracy = correct / total if total else 0.0
    return {
        "accuracy": accuracy,
        "count": total,
        "correct": correct,
        "missing": missing,
        "skipped_open": skipped_open,
    }


def _read_records(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return read_csv_records(path)
    return read_jsonl(path)


def evaluate_closed_qa(
    gold_path: str | Path,
    pred_path: str | Path,
    *,
    on_open: str = "skip",
) -> dict[str, float | int]:
    """Read gold/prediction files and evaluate closed QA accuracy."""
    return evaluate_accuracy(_read_records(gold_path), _read_records(pred_path), on_open=on_open)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate closed QA predictions.")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--on-open", choices=["skip", "error"], default="skip")
    args = parser.parse_args()

    result = evaluate_closed_qa(args.gold, args.pred, on_open=args.on_open)
    print(
        "accuracy={accuracy:.4f} count={count} correct={correct} "
        "missing={missing} skipped_open={skipped_open}".format(**result)
    )


if __name__ == "__main__":
    main()
