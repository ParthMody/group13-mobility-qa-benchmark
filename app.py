"""
Task 3 — Two-Period Change Detection: benchmark module.
"""
import os
import json
from pathlib import Path
 
from flask import (Flask, render_template, jsonify, abort, request, redirect, url_for)
 
BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
 
# Task registry. Each task becomes a tab. Owners/copy are display-only.
TASKS = [
    {"id": "task1", "num": 1, "short": "Next-POI Category", "name": "Next-POI Category QA",
     "owner": "Dhanesh", "kind": "Multiple choice",
     "tests": "Given a recent check-in sequence, pick the most likely next POI category.",
     "scoring": "Accuracy, F1, and top-k."},
    {"id": "task2", "num": 2, "short": "Weekday vs Weekend", "name": "Weekday vs Weekend QA",
     "owner": "Chinmoy", "kind": "Binary",
     "tests": "Given a day of check-ins, judge whether it looks like a weekday or a weekend.",
     "scoring": "Accuracy and F1."},
    {"id": "task3", "num": 3, "short": "Change Detection", "name": "Two-Period Change Detection QA",
     "owner": "Parth", "kind": "Open answer + label",
     "tests": "Given a city or user's check-ins in 2013 and 2018, describe how the visitation pattern changed.",
     "scoring": "Change-label accuracy plus an open-answer rubric."},
    {"id": "task4", "num": 4, "short": "Zero-Shot Reasoning", "name": "Zero-Shot POI Reasoning QA",
     "owner": "Hanju", "kind": "Open answer",
     "tests": "Given a scenario, rank the most likely next POIs and explain the reasoning.",
     "scoring": "Reasoning quality, with an optional ranking metric."},
    {"id": "task5", "num": 5, "short": "Preference Shift", "name": "User Preference Shift QA",
     "owner": "Jiayi", "kind": "Multiple choice",
     "tests": "Given a check-in sequence and a short preference update, pick the next category that fits the new preference.",
     "scoring": "Accuracy."},
]
TASK_BY_ID = {t["id"]: t for t in TASKS}
 
 
def load_items(task_id):
    path = DATA_DIR / f"{task_id}_items.jsonl"
    items = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    return items
 
 
ITEMS = {t["id"]: load_items(t["id"]) for t in TASKS}
for t in TASKS:
    t["count"] = len(ITEMS[t["id"]])
    t["status"] = "live" if t["count"] else "planned"
    # Task 3 has a bespoke change-detection view; any other task renders generically.
    t["view"] = "change" if t["id"] == "task3" else "generic"
 
app = Flask(__name__)
 
 
def nav(active_id, section=None):
    return {"tasks": TASKS, "active_id": active_id, "task": TASK_BY_ID.get(active_id), "section": section}
 
 
def change_rows(item):
    """Build the 2013-vs-2018 category comparison for a change-detection item."""
    periods = item["context_sequence"]
    a, b = periods[0].get("category_mix", {}), periods[1].get("category_mix", {})
    pa, pb = periods[0].get("period", "A"), periods[1].get("period", "B")
    seen, rows = set(), []
    for c in list(a.keys()) + list(b.keys()):
        if c in seen:
            continue
        seen.add(c)
        av, bv = a.get(c, 0), b.get(c, 0)
        rows.append({"cat": c, "a": av, "b": bv, "delta": round(bv - av, 3)})
    rows.sort(key=lambda r: -r["b"])
    maxv = max([max(r["a"], r["b"]) for r in rows] + [0.01])
    return pa, pb, rows, maxv
 
 
# ---------- pages ----------
@app.route("/")
def home():
    live = next((t for t in TASKS if t["status"] == "live"), TASKS[0])
    return redirect(url_for("task_view", tid=live["id"]))
 
 
@app.route("/task/<tid>")
def task_view(tid):
    t = TASK_BY_ID.get(tid)
    if not t:
        abort(404)
    if t["status"] != "live":
        return render_template("placeholder.html", **nav(tid))
    tmpl = "items.html" if t["view"] == "change" else "generic_items.html"
    return render_template(tmpl, items=ITEMS[tid], **nav(tid, "items"))
 
 
@app.route("/task/<tid>/item/<qid>")
def item_view(tid, qid):
    t = TASK_BY_ID.get(tid)
    if not t or t["status"] != "live":
        abort(404)
    item = next((it for it in ITEMS[tid] if it["question_id"] == qid), None)
    if not item:
        abort(404)
    raw = json.dumps(item, indent=2, ensure_ascii=False)
    if t["view"] == "change":
        pa, pb, rows, maxv = change_rows(item)
        return render_template("item.html", item=item, pa=pa, pb=pb, rows=rows, maxv=maxv,
                               raw=raw, **nav(tid, "items"))
    return render_template("generic_item.html", item=item, raw=raw, **nav(tid, "items"))
 
 
@app.route("/task/<tid>/evaluate")
def evaluate_view(tid):
    t = TASK_BY_ID.get(tid)
    if not t or t["status"] != "live":
        abort(404)
    from src.baseline import majority_baseline
    from src.evaluate import evaluate_all
    items = ITEMS[tid]
    preds = majority_baseline(items)
    report = evaluate_all(items, preds)
    by_id = {it["question_id"]: it for it in items}
    return render_template("evaluate.html", report=report, preds=preds, items_by_id=by_id,
                           **nav(tid, "evaluate"))
 
 
# ---------- JSON API ----------
@app.route("/api/<tid>/items")
def api_items(tid):
    if tid not in TASK_BY_ID:
        abort(404)
    return jsonify(ITEMS[tid])
 
 
@app.route("/api/<tid>/items/<qid>")
def api_item(tid, qid):
    if tid not in TASK_BY_ID:
        abort(404)
    it = next((x for x in ITEMS[tid] if x["question_id"] == qid), None)
    if not it:
        abort(404)
    return jsonify(it)
 
 
@app.route("/api/<tid>/evaluate", methods=["POST"])
def api_evaluate(tid):
    if tid not in TASK_BY_ID:
        abort(404)
    from src.evaluate import evaluate_all
    payload = request.get_json(force=True, silent=True) or {}
    return jsonify(evaluate_all(ITEMS[tid], payload.get("predictions", {})))
 
 
@app.route("/health")
def health():
    return {"status": "ok", "tasks": {t["id"]: t["count"] for t in TASKS}}
 
 
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)