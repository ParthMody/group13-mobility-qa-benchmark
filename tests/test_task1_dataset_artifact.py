import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mobility_qa.io import read_csv_records, read_jsonl  # noqa: E402
from mobility_qa.schema import validate_qa_records  # noqa: E402
from mobility_qa.tasks.build_task1_dataset import BROAD_CATEGORIES  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data/benchmark/task1_massive_steps"
CSV_PATH = DATASET_DIR / "task1_massive_steps_test.csv"
JSONL_PATH = DATASET_DIR / "task1_massive_steps_test.jsonl"
SUMMARY_PATH = DATASET_DIR / "task1_massive_steps_test_summary.json"


def test_generated_dataset_files_match_and_validate():
    csv_records = read_csv_records(CSV_PATH)
    jsonl_records = read_jsonl(JSONL_PATH)

    assert len(jsonl_records) == 2832
    assert csv_records == jsonl_records
    assert validate_qa_records(jsonl_records) is True


def test_generated_dataset_has_expected_coverage_and_no_venue_leakage():
    records = read_jsonl(JSONL_PATH)

    assert len({record["city"] for record in records}) == 15
    assert {record["correct_answer"] for record in records} == set(BROAD_CATEGORIES)
    assert {record["difficulty"] for record in records} == {"easy", "medium", "hard"}
    for record in records:
        assert len(record["options"]) == 4
        assert record["verification_status"] == (
            "verified_massive_steps_taxonomy_mapping"
        )
        assert "POI id" not in record["context"]
        assert "venue_id" not in record["context"]


def test_generated_summary_matches_artifact():
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    assert summary["total_records"] == 2832
    assert len(summary["cities"]) == 15
    assert len(summary["source_files"]) == 15
    assert all(
        len(source["sha256"]) == 64 for source in summary["source_files"].values()
    )
    assert len(summary["taxonomy_sha256"]) == 64
    assert sum(summary["category_counts"].values()) == 2832
    assert sum(summary["difficulty_counts"].values()) == 2832
