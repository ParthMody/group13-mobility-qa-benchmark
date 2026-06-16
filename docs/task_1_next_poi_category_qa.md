# Task 1: Next-POI Category QA

## Goal

Given a user's previous POI check-in sequence, predict the most likely next POI category as a question-answering task.

## Input

Each example should contain:

- `question_id`: unique identifier for the QA example
- `user_id`: anonymised user identifier
- `history`: ordered previous POI category sequence
- `question`: natural-language question
- `choices`: candidate answer categories
- `answer`: correct next POI category

## Output

The model should return one category from the candidate choices.

## Example

```csv
question_id,user_id,history,question,choices,answer
task1_001,user_001,"home > transit_station > office","What is the most likely next POI category?","restaurant|cafe|gym|home",restaurant
```

## Notes

- Start with category-level prediction, not exact POI prediction.
- Keep histories short enough to read in a QA prompt.
- Avoid including personally identifying information.
- Store raw source data in `data/raw/` and generated QA examples in `data/processed/`.
