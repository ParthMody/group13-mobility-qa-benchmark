# Task 1 dataset splits

These splits apply only to `task1_next_poi_category`. Records from Tasks 2, 3,
4, and 5 are not read or included.

## Purpose

The initial Task 1 set contains at most 400 valid records. It supports the first
benchmark release, app integration, baseline testing, marking progress, and
first-round experiments.

The extended set contains every valid Task 1 record left after the initial set.
It is reserved for later, larger evaluations and benchmark expansion.

The split process never creates records to reach 400. When fewer than 400 valid
records are available, the initial set contains all of them and the extended
set is empty. This shortage is recorded in the manifest.

## Approved inputs

The build script reads only these patterns:

```text
data/examples/task1_*.csv
data/examples/task1_*.jsonl
data/processed/task1_*.csv
data/processed/task1_*.jsonl
```

Generated combined files, benchmark outputs, and files for other tasks are
excluded. Legacy Task 1 field names are normalized to the shared template:

```text
question_id
task
city
context
answer_type
options
correct_answer
reasoning
difficulty
source_dataset
verification_status
```

Invalid records are excluded. Identical CSV and JSONL representations with the
same `question_id` are included once. Conflicting records with the same
`question_id` stop the build with a clear error.

## Reproducible selection

When more than 400 valid records are available, the initial set uses
deterministic stratified selection across city, difficulty, and correct-answer
POI category where possible. Seed `5925` provides deterministic tie-breaking.
The process preserves representation from available cities and difficulty
levels and limits answer-category concentration where the data permits.

The same valid input and seed produce the same initial and extended memberships.
No question ID appears in both sets, and original question IDs are never
changed.

## Build and split

From the repository root:

```bash
python scripts/build_task1_dataset.py

python scripts/create_task1_splits.py \
  --input data/processed/task1_all_records.jsonl \
  --initial-size 400 \
  --seed 5925
```

The combined normalized source is written to:

```text
data/processed/task1_all_records.csv
data/processed/task1_all_records.jsonl
```

The release files are:

```text
data/benchmark/task1/task1_initial_400.csv
data/benchmark/task1/task1_initial_400.jsonl
data/benchmark/task1/task1_extended.csv
data/benchmark/task1/task1_extended.jsonl
```

The manifest is
`data/benchmark/task1/task1_split_manifest.json`. It records the requested and
actual split sizes, seed, strategy, source, exclusions, and total/initial/
extended distributions by city, difficulty, and answer category.

Rejected invalid inputs and removed duplicate representations are listed in
`data/benchmark/task1/task1_rejected_records.csv`.
