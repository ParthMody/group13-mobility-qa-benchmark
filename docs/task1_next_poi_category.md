# Task 1: Next-POI Category QA

Task 1 is Dhanesh's closed multiple-choice task. Given a short sequence of
recent check-ins and a target time, the model predicts the next **broad POI
category** from the supplied options. It does not predict a venue name or POI
ID.

## Boundary With Task 4

Task 1 returns one category from a fixed option list. Task 4 is an open
reasoning task whose response explains mobility behaviour in free text. The
short `reasoning` field in a Task 1 gold record documents why its label is
plausible; it does not turn Task 1 into an open-response task.

## Datasets

The five records in `data/examples/task1_dhanesh_5_questions.csv` and its JSONL
equivalent are draft manual examples used to check the benchmark format. They
follow the Massive-STEPS-style benchmark direction and therefore use
`source_dataset = massive_steps`, but they are not extracted Massive-STEPS
rows. Every record is marked `verification_status = draft_manual_example`.

The main benchmark is now generated from the official Massive-STEPS test splits:

```text
data/benchmark/task1_massive_steps/task1_massive_steps_test.csv
data/benchmark/task1_massive_steps/task1_massive_steps_test.jsonl
```

It contains 2,832 verified questions across all 15 released cities. Target
labels are derived from source `venue_category_id` values through the historical
Foursquare top-level taxonomy. See `docs/task1_dataset_build.md` for the full
methodology and limitations.

## Evaluation

Predictions contain `question_id` and `prediction`. Task 1 uses exact-match
accuracy after case and whitespace normalisation.
