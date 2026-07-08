# Shared QA Format

Parth's current benchmark template uses these fields:

| Field | Meaning |
| --- | --- |
| `question_id` | Unique question identifier |
| `task` | Benchmark task name |
| `city` | City associated with the mobility context |
| `context` | Check-in history and target time shown to the model |
| `answer_type` | `multiple choice` or `written` |
| `options` | Candidate answers for a multiple-choice item |
| `correct_answer` | Gold answer |
| `reasoning` | Short justification for the gold answer |
| `difficulty` | `easy`, `medium`, or `hard` |
| `source_dataset` | Dataset direction; currently `massive_steps` |
| `verification_status` | Provenance or review state of the record |

Current Task 1 verification values are:

- `draft_manual_example`: manually authored format-check record
- `verified_massive_steps_taxonomy_mapping`: target taken from an official
  Massive-STEPS trail and mapped by Foursquare category ID

## Answer Types

- `multiple choice`: `options` must be non-empty and `correct_answer` must
  exactly match one option.
- `written`: `options` may be empty and `correct_answer` contains the written
  reference answer.

Task 1 uses `multiple choice`.

## CSV And JSONL

CSV uses the columns in the order shown above. Options use `|` as the separator:

```text
Office | Park | Nightlife | Hospital
```

JSONL stores one JSON object per line and represents `options` as a JSON array.
The CSV and JSONL files contain the same records and field labels.
