# Evaluation Plan

Closed tasks use accuracy first. This applies to Tasks 1, 2, and 5 where
`metadata.answer_type = "closed"`.

Open tasks use rubric-based evaluation later. This applies to Tasks 3 and 4
where `metadata.answer_type = "open"`.

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
