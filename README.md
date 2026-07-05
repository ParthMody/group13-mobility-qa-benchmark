# Waypoint Analytics — Mobility & POI Reasoning Benchmark

A multi-task benchmark for mobility and Point-of-Interest (POI) reasoning, built on the
[Massive-STEPS](https://github.com/cruiseresearchgroup/Massive-STEPS) check-in dataset
(2013 and 2018, multiple cities). A deployed web application hosts all five tasks, their
items, and the evaluation.

**Live app:** https://waypoint-analytics.up.railway.app/

Group 13 · DATA5925

## Tasks

| # | Task | Answer type | Status |
|---|------|-------------|--------|
| 1 | Next-POI Category QA | multiple choice | QA items authored |
| 2 | Weekday vs Weekend QA | binary | QA items authored |
| 3 | Two-Period Change Detection QA | open + label | **evaluated** (baseline + zero-shot LLM) |
| 4 | Zero-Shot POI Reasoning QA | open | QA items authored |
| 5 | User Preference Shift QA | multiple choice | QA items authored |

Task 3 is the worked reference: real multi-city items, a majority baseline, and a
zero-shot LLM track (both predict the same change label; the LLM's written reasoning is
scored separately against a rubric).

## Setup

```bash
pip install -r requirements.txt
python app.py            # http://localhost:8000
```

The app loads each task's items from `data/<task_id>_items.jsonl` into memory (no
database). A task tab goes live automatically once its items file is present.

## Task 3 pipeline (reproduce the results)

```bash
# 1. pull a city from Massive-STEPS and flatten to check-ins
python src/fetch_city.py --city Sydney --out data/sydney_checkins.csv

# 2. check the 2013/2018 user overlap (decides the unit)
python src/user_overlap.py --checkins data/sydney_checkins.csv

# 3. build city-level items (repeat per city; appends)
python src/build_items.py --checkins data/sydney_checkins.csv --city Sydney --unit city

# 4. zero-shot LLM track (set an API key in the environment first)
python src/llm_zeroshot.py --model gemini-2.5-flash

# 5. baseline + combined scoreboard -> data/results.json
python src/ml_baseline.py
python src/scoreboard.py
```

API keys are read from the environment (never committed).

## Repository structure

```text
app.py                  Flask app: task registry, routes, evaluation view
templates/              Server-rendered views (base, items, item, evaluate, generic, placeholder)
static/                 Brand assets
data/
  *_items.jsonl         Per-task QA items loaded by the app
  results.json          Task 3 scoreboard
  src/                  Raw QA sources (Notion exports) for conversion
src/
  fetch_city.py         Pull + flatten a Massive-STEPS city
  user_overlap.py       2013 vs 2018 user-overlap check
  build_items.py        Build Task 3 change-detection items
  llm_zeroshot.py       Zero-shot LLM track + reasoning judge
  ml_baseline.py        Majority / traditional-ML baseline
  scoreboard.py         Combine tracks -> results.json
  convert_qa.py         Convert authored QA (Notion/JSONL) into the item schema
  evaluate.py           Scoring helpers
docs/                   Task and dataset documentation
tests/                  Unit tests
```

## Status

Task 3 is evaluated end-to-end across multiple cities. Tasks 1, 2, 4 and 5 are populated
with authored QA items; their scoring harnesses are not yet wired. Deployed on Railway.
