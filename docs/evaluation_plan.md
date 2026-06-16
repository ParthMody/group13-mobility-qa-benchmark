# Evaluation Plan

## Initial Metric

Use exact-match accuracy for Task 1:

```text
accuracy = number of correct predicted categories / total examples
```

Predictions should be compared against the `answer` field.

## Expected Prediction Format

Use a simple CSV or JSONL file with:

- `question_id`
- `prediction`

## Evaluation Rules

- Match predictions to gold examples by `question_id`.
- Compare normalised category strings.
- Count missing predictions as incorrect.
- Report total examples, correct examples, missing predictions, and accuracy.

## Future Metrics

Possible later additions:

- top-k accuracy
- per-category accuracy
- confusion matrix
- split-level reporting for train/dev/test

These should only be added after the basic Task 1 generator and exact-match evaluator are working.
