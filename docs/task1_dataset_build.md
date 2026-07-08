# Task 1 Dataset Build

## Source

The generated Task 1 benchmark uses the official Massive-STEPS city datasets:

- Repository: <https://github.com/cruiseresearchgroup/Massive-STEPS>
- Dataset collection: <https://huggingface.co/collections/CRUISEResearchGroup/massive-steps-point-of-interest-check-in-dataset>
- Paper: <https://arxiv.org/abs/2505.11239>
- License: Apache-2.0

The builder downloads the `tabular` test-split CSV for all 15 cities. Raw
downloads remain under `data/raw/massive_steps/` and are excluded from Git.

## Label Construction

Massive-STEPS provides ordered check-ins grouped by `trail_id`. For every
eligible trail:

1. The last check-in is held out as the prediction target.
2. Up to six preceding check-ins form the context.
3. The target's `venue_category_id` is mapped through the historical
   Foursquare category tree to one of its 10 top-level categories.
4. Trails containing unmapped categories in the relevant source sequence are
   skipped rather than assigned an inferred label.

The category tree snapshot is downloaded from
<https://gist.github.com/missinglink/a0344050d3e2b52256d7>, which archived the
legacy Foursquare category endpoint used by the 2012-2018 source check-ins.

## Question Construction

Prompts include timestamps and fine-grained category names. They exclude user
IDs, venue names, addresses, coordinates, venue IDs, and POI IDs.

Each item has four options. The three distractors prioritize broad categories
that actually followed the last observed category elsewhere in the same city's
source split. Remaining slots use city-level category frequency. Option order
is deterministically shuffled with seed `5925`.

Difficulty is assigned without model judgments:

- `easy`: target broad category equals the last observed broad category
- `medium`: target appears earlier in the retained history
- `hard`: target does not appear in the retained history

## Generated Artifact

The committed test benchmark contains 2,832 records across all 15 cities:

- 649 easy
- 607 medium
- 1,576 hard

All 10 broad categories are represented. Each city contributes at most 200
balanced records; Beijing contributes all 32 eligible mapped trajectories.

Files:

```text
data/benchmark/task1_massive_steps/task1_massive_steps_test.csv
data/benchmark/task1_massive_steps/task1_massive_steps_test.jsonl
data/benchmark/task1_massive_steps/task1_massive_steps_test_summary.json
```

## Rebuild

```bash
python3 -m src.tasks.build_task1_dataset \
  --download \
  --split test \
  --per-city 200
```

The build is deterministic for the same upstream files, taxonomy snapshot,
city list, and seed. The summary records the download URLs and SHA-256 checksum
of every source CSV and the taxonomy file so the exact build inputs can be
audited later.
