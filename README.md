# Waypoint Analytics — Mobility & POI Reasoning Benchmark

A multi-task benchmark for mobility and Point-of-Interest (POI) reasoning, built on the
[Massive-STEPS](https://github.com/cruiseresearchgroup/Massive-STEPS) check-in dataset
(2013 and 2018, multiple cities). A deployed web application hosts all five tasks, their
items, and per-task evaluation.

**Live app:** https://waypoint-analytics.up.railway.app/

Group 13 · DATA5925

## Tasks

All tasks are evaluated with a zero-shot LLM (Gemini 2.5-flash) so results are comparable.
Multiple-choice tasks are scored by exact-match accuracy and macro-F1; open-answer tasks
by an LLM judge against a per-item rubric.

| # | Task | Answer type | Status |
|---|------|-------------|--------|
| 1 | Next-POI Category QA | multiple choice | evaluated (accuracy) |
| 2 | Weekday vs Weekend QA | binary | evaluated (accuracy) |
| 3 | Two-Period Change Detection QA | open + label | evaluated (baseline + zero-shot LLM + reasoning) |
| 4 | Zero-Shot POI Reasoning QA | open | evaluated (reasoning judge) |
| 5 | User Preference Shift QA | multiple choice | evaluated (accuracy) · items being scaled |

Task 3 is the worked reference: real multi-city items built from Massive-STEPS, a majority
baseline, and a zero-shot LLM track (both predict the same change label; the LLM's written
reasoning is scored separately against a rubric). Each task's numbers are under its
**Evaluate** tab in the app.

## Setup

```bash
pip install -r requirements.txt
python app.py            # http://localhost:8000
```

The app loads each task's items from `data/<task_id>_items.jsonl` into memory (no
database). A task tab goes live once its items file is present, and shows an Evaluate tab
once its `data/<task_id>_results.json` exists.

## Task 3 pipeline (reproduce the results)

```bash
# 1. pull a city from Massive-STEPS and flatten to check-ins
python src/fetch_city.py --city Sydney --out data/sydney_checkins.csv

# 2. check the 2013/2018 user overlap (decides the unit)
python src/user_overlap.py --checkins data/sydney_checkins.csv

# 3. build city-level items (repeat per city; appends)
python src/build_items.py --checkins data/sydney_checkins.csv --city Sydney --unit city

# 4. zero-shot LLM track + baseline + scoreboard (set an API key first)
python src/llm_zeroshot.py --model gemini-2.5-flash
python src/ml_baseline.py
python src/scoreboard.py          # -> data/results.json
```

## Evaluate the other tasks (1, 2, 4, 5)

```bash
# scores MCQ tasks by accuracy, open tasks by the LLM judge
python -m src.eval_task --task task1 --model gemini-2.5-flash
python -m src.eval_task --task task2 --model gemini-2.5-flash
python -m src.eval_task --task task4 --model gemini-2.5-flash
python -m src.eval_task --task task5 --model gemini-2.5-flash
# writes data/<task>_results.json, shown on each task's Evaluate tab
```

API keys are read from the environment (`GEMINI_API_KEY`), never committed.

## Repository structure

```text
app.py                  Flask app: task registry, routes, evaluation views
templates/              Server-rendered views (base, items, item, evaluate,
                        evaluate_generic, generic_items/item, placeholder)
static/                 Brand assets
data/
  *_items.jsonl         Per-task QA items loaded by the app
  results.json          Task 3 scoreboard
  *_results.json        Per-task evaluation results (tasks 1, 2, 4, 5)
  src/                  Raw QA sources (Notion exports) for conversion
src/
  fetch_city.py         Pull + flatten a Massive-STEPS city
  user_overlap.py       2013 vs 2018 user-overlap check
  build_items.py        Build Task 3 change-detection items (theme-based labelling)
  llm_zeroshot.py       Task 3 zero-shot LLM track + reasoning judge (Gemini/Anthropic)
  eval_task.py          Generalised per-task evaluation (MCQ accuracy / open judge)
  ml_baseline.py        Majority / traditional-ML baseline
  scoreboard.py         Combine Task 3 tracks -> results.json
  convert_qa.py         Convert authored QA (Notion/JSONL) into the item schema
  list_models.py        List models available to your API key
  evaluate.py           Scoring helpers
docs/                   Task and dataset documentation
tests/                  Unit tests
```

## Contributing

The `main` branch auto-deploys to Railway. Please branch and open a pull request rather
than committing directly to `main`, and run `python app.py` locally first to confirm it
starts cleanly.

## Status

All five tasks are evaluated with a consistent zero-shot LLM (Gemini 2.5-flash) and shown
in the app. Task 3 is the fully worked reference (multi-city items, baseline, zero-shot,
reasoning). Task 5's items are being scaled (more cities, harder items). Deployed on Railway.
