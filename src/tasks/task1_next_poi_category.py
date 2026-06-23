"""CLI wrapper for Task 1 example generation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mobility_qa.tasks.task1_next_poi_category import (  # noqa: E402,F401
    build_task1_question,
    create_task1_record,
    export_task1_examples,
    main,
)


if __name__ == "__main__":
    main()
