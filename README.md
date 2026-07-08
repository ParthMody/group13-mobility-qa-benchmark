# DATA5925 Group 13 Mobility QA Benchmark

This repository contains a lightweight student benchmark for mobility and
Point-of-Interest (POI) question answering. All current examples use the shared
QA format and `source_dataset = "massive_steps"`.

## Dhanesh Scope

Dhanesh owns:

- Task 1: Next POI Category QA
- shared benchmark format
- closed-QA scoring
- Task 1 setup and draft examples

Parth owns Flask, task consolidation, and deployment, so this repo code stays
focused on data format, examples, validation, and evaluation utilities.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Tests

```bash
pytest
```

## Generate Task 1 Examples

```bash
python3 -m src.tasks.task1_next_poi_category \
  --output-csv data/examples/task1_dhanesh_5_questions.csv \
  --output-jsonl data/examples/task1_dhanesh_5_questions.jsonl
```

The five generated examples are manual category-level records for format checking.
They are not extracted Massive-STEPS rows and use
`verification_status = "draft_manual_example"`.

## Build The Dataset-Backed Task 1 Benchmark

The main Task 1 benchmark contains 2,832 questions from the official
Massive-STEPS test splits across all 15 cities. Raw check-ins are downloaded to
the ignored `data/raw/` directory; generated benchmark files are written to
`data/benchmark/task1_massive_steps/`.

```bash
python3 -m src.tasks.build_task1_dataset \
  --download \
  --split test \
  --per-city 200
```

The source is Apache-2.0 licensed. The builder derives broad category labels
from Foursquare category IDs, excludes venue and user identifiers from prompts,
uses transition-based distractors, and emits a provenance summary. Full method
details are in `docs/task1_dataset_build.md`.

## Evaluate Closed QA Predictions

Prediction files should contain `question_id` and `prediction`. JSONL is the
default format; CSV is also supported.

```bash
python3 -m src.evaluation.evaluate_closed_qa \
  --gold data/examples/task1_dhanesh_5_questions.jsonl \
  --pred results/task1_predictions.jsonl
```

## Repository Structure

```text
data/
  benchmark/    Generated, versioned benchmark datasets
  raw/          Raw Massive-STEPS files, not committed by default
  processed/    Generated benchmark outputs, not committed by default
  examples/     Small example QA records
  schema/       JSON schema documentation
docs/           Shared format, task, and evaluation notes
src/
  mobility_qa/  Validation, IO, task, and evaluation package
  tasks/        CLI wrappers for task utilities
  evaluation/   CLI wrappers for evaluation utilities
tests/          Pytest coverage
```
