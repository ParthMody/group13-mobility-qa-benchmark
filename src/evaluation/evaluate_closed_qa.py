"""CLI entry point for Parth-format multiple-choice QA evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mobility_qa.evaluation.evaluate_closed_qa import (  # noqa: E402,F401
    evaluate_accuracy,
    evaluate_closed_qa,
    main,
    normalize_answer,
)

__all__ = ["normalize_answer", "evaluate_accuracy", "evaluate_closed_qa", "main"]


if __name__ == "__main__":
    main()
