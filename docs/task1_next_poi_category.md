# Task 1: Next-POI Category QA

## Goal

Given a recent sequence of timestamped POI categories, select the broad category
of the user's next check-in from four choices. Task 1 predicts a category, not a
specific venue or POI identifier.

Task 1 is closed multiple-choice QA. This separates it from Task 4, which asks
for an open prediction and written reasoning about likely future behaviour.

## Data Source

The five evaluation items in `data/task1_items.jsonl` are a fixed subset of the
official Massive-STEPS test splits. They replace the previous manual draft
scenarios while retaining a small item count for comparison with the other
prototype tasks.

For each item:

1. The final check-in in a Massive-STEPS trail is held out as the target.
2. Earlier check-ins provide the sequence shown to the model.
3. The held-out `venue_category_id` is mapped through the historical Foursquare
   taxonomy to one broad category.
4. The broad category becomes the answer and three other broad categories become
   distractors.

The selected items cover Jakarta, Sydney, Melbourne, Tokyo, and New York; five
different target categories; and easy, medium, and hard difficulty levels.

## Provenance

Each item retains its source split, trail ID, held-out fine category, and target
Foursquare category ID in `metadata`. Verified items use:

```text
source_dataset = massive_steps
metadata.verification_status = verified_massive_steps_taxonomy_mapping
```

User and venue identifiers are not exposed to the model. `user_id` is therefore
stored as `ANONYMISED`, and context entries contain only category and timestamp.

## Difficulty

- `easy`: the target broad category repeats the last observed broad category.
- `medium`: the target broad category appears earlier in the retained history.
- `hard`: the target broad category is absent from the retained history.

## Evaluation

Task 1 is evaluated by exact match against `answer`. The main metric is accuracy,
with macro-F1 reported across the answer classes. Run:

```bash
python -m src.eval_task --task task1 --model gemini-2.5-flash
```

This writes `data/task1_results.json`. The previous result file was removed when
the grounded items replaced the manual questions because its predictions and
question IDs no longer matched the evaluation set.

## Limitation

Five items are enough for a transparent pilot but not for a statistically stable
estimate of next-POI performance. One answer changes accuracy by 20 percentage
points, and the subset cannot represent every Massive-STEPS city or category.
