"""CLI entry point for the Massive-STEPS Task 1 dataset builder."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mobility_qa.tasks.build_task1_dataset import main  # noqa: E402


if __name__ == "__main__":
    main()
