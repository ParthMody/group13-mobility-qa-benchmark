# Evaluation Plan

Closed tasks use accuracy first. This applies to records where
`answer_type = "multiple choice"`.

Written tasks use rubric-based evaluation later. These records use
`answer_type = "written"`.

## Task 1

Task 1 current metric is exact-match accuracy:

```text
accuracy = correct closed predictions / total closed examples
```

Predictions should include:

```text
question_id, prediction
```

The evaluator matches by `question_id`, normalises whitespace and case, and
counts missing predictions as incorrect.

## Future Metrics

- macro-F1 for category balance
- top-k accuracy for ranked category predictions
- per-category accuracy
- confusion matrix
- split-level reporting for train/dev/test
