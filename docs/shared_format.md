# Shared QA Format

All benchmark tasks use the same QA record shape so Parth's app can load one
format across closed and open tasks.

## Fields

- `question_id`: unique question identifier
- `task`: task name, such as `task1_next_poi_category`
- `city`: city for the check-in sequence
- `user_id`: anonymised user identifier
- `context_sequence`: ordered previous POI/check-in context
- `target_time`: time associated with the next POI target
- `question`: natural-language prompt
- `choices`: candidate answers for closed tasks; empty list allowed for open tasks
- `answer`: gold answer
- `rationale`: short human-readable reason for the answer
- `source_dataset`: currently `massive_steps`
- `metadata`: task and evaluation metadata

## Metadata

Required metadata fields:

- `answer_type`: `closed` or `open`
- `eval_mode`: evaluation mode, for example `classification`
- `difficulty`: `easy`, `medium`, or `hard`

Tasks 1, 2, and 5 use `answer_type = "closed"`.
Tasks 3 and 4 use `answer_type = "open"`.

Closed records must have non-empty `choices`, and `answer` must exactly match one
choice. Open records may have empty `choices`, but `answer` must be a non-empty
written answer.

## CSV And JSONL

JSONL stores one full JSON record per line.

CSV uses the same columns. The `choices` column is pipe-separated, for example:

```text
Office|Cafe|Park|Restaurant
```

The `metadata` column is JSON text. Complex columns such as `context_sequence`
are also stored as JSON text in CSV.
