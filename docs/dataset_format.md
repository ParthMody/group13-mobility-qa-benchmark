# Dataset Format

## Raw Check-In CSV

Raw check-in data should be stored in `data/raw/` and should not be modified in place.

Required columns:

- `user_id`
- `venue_id`
- `venue_category`
- `timestamp`
- `venue_city`
- `latitude`
- `longitude`

## QA JSONL

Generated QA examples should be stored as JSONL records.

Required fields:

- `question_id`
- `task`
- `city`
- `user_id`
- `context_sequence`
- `target_time`
- `question`
- `choices`
- `answer`
- `rationale`
- `source_dataset`
- `metadata`

The `choices` field should be a list of candidate POI categories, and `answer` should be one of those choices.
