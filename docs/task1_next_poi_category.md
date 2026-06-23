# Task 1: Next POI Category QA

Task 1 asks a model to predict the next broad POI category from a user's recent
check-in sequence.

The task is closed multiple-choice. The model must choose one category from the
provided `choices`; it should not predict a specific venue or POI ID.

## Boundary With Task 4

Task 1 is category classification. It asks only for the next broad POI category.

Task 4 is open reasoning and explanation. Do not duplicate Task 4 by asking for a
free-form route explanation, behavioural explanation, or detailed venue-level
reasoning in Task 1.

## Data Source

The current source dataset is Massive-STEPS only. Use
`source_dataset = "massive_steps"` for current examples.

Because raw Massive-STEPS POI ID to category mapping is not committed in this
repo, current example rows are marked in metadata with:

```json
{"verification_status": "draft_category_example_pending_raw_mapping"}
```

## Shared Sheet Columns

Fill these columns:

```text
question_id, task, city, user_id, context_sequence, target_time, question,
choices, answer, rationale, source_dataset, metadata
```

## Example Row

```csv
task1_dhanesh_001,task1_next_poi_category,Tokyo,ms_user_0001,"[{...}]",2024-03-04T13:10:00,"Given the user's recent check-ins in Tokyo, what broad POI category is the most likely next stop at 2024-03-04T13:10:00?",Office|Cafe|Park|Restaurant,Office,"The weekday sequence follows a commute from residence through transport to work, so the next broad category remains office.",massive_steps,"{""answer_type"": ""closed"", ""eval_mode"": ""classification"", ""difficulty"": ""easy"", ""verification_status"": ""draft_category_example_pending_raw_mapping""}"
```

## Evaluation

The current Task 1 metric is exact-match accuracy after simple answer
normalisation. Macro-F1 and top-k accuracy can be added later once there are
larger labelled splits.
