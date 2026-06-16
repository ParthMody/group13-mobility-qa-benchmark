# QA-Based Mobility and POI Behaviour Benchmark

This repository contains a lightweight university benchmark for mobility and Point-of-Interest (POI) behaviour question answering.

The current scaffold supports **Task 1: Next-POI Category QA**. Given a user's previous POI check-in sequence, the model must answer the most likely next POI category.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Tests

```bash
pytest
```

## Repository Structure

```text
data/
  raw/          Original source data, not committed by default
  processed/    Generated benchmark outputs, not committed by default
  examples/     Small synthetic examples for development
  schema/       JSON schema documentation for QA records
docs/           Dataset and task documentation
src/
  mobility_qa/  Python package for schema, IO, tasks, and evaluation
tests/          Unit tests
```

## Current Status

Task 1 scaffold only:

- check-in CSV column validation
- QA JSONL record validation
- simple JSONL and CSV IO helpers
- synthetic example data
- no model training
- no heavy ML code
- no external APIs
