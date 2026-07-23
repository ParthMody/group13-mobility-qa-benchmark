import os
import json
import re
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
def model_rows():
    """Multi-model comparison for Task 3.

    Pairs each label run (data/preds_llm*.jsonl, written by llm_zeroshot.py) with
    its probe run (data/task3probe_results*.json, written by eval_task.py) using
    the filename suffix: preds_llm_lite.jsonl <-> task3probe_results_lite.json.
    The model name is read from the probe file, which records it. Returns [] until
    at least two label runs exist, so the block stays hidden on a single-model run.
    """
    items = ITEMS.get("task3") or []
    if not items:
        return []
    try:
        gold = {it["question_id"]: it["metadata"]["change_label"] for it in items}
    except KeyError:
        return []

    rows = []
    for p in sorted(DATA_DIR.glob("preds_llm*.jsonl")):
        if p.stem.endswith("_cot"):
            continue
        suffix = p.stem[len("preds_llm"):]          # "" or "_lite"
        data = {}
        try:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        data[r["question_id"]] = r
        except Exception:
            continue
        if not data:
            continue
        pred = {q: data.get(q, {}).get("change_label", "invalid") for q in gold}
        label_acc = sum(1 for q in gold if gold[q] == pred[q]) / len(gold)
        rs = [data[q]["reasoning_score"] for q in data if "reasoning_score" in data[q]]
        reasoning = round(sum(rs) / len(rs), 3) if rs else None

        probe_acc, probe_n, model = None, None, None
        pp = DATA_DIR / f"task3probe_results{suffix}.json"
        if pp.exists():
            try:
                pr = json.load(open(pp, encoding="utf-8"))
                probe_acc, probe_n, model = pr.get("accuracy"), pr.get("n_items"), pr.get("model")
            except Exception:
                pass
        rows.append({"model": model or (suffix.lstrip("_") or "default"),
                     "label_acc": round(label_acc, 3), "reasoning": reasoning,
                     "probe_acc": probe_acc, "probe_n": probe_n,
                     "n_label": len(gold)})
    if len(rows) < 2:
        return []
    return sorted(rows, key=lambda r: -(r["probe_acc"] or 0))


def analysis_files():
    """(city, n_units, path) for every neighbourhood analysis present on disk."""
    paths = sorted(DATA_DIR.glob("hood_*.json"))
    default = DATA_DIR / "task3_neighbourhood_analysis.json"
    if default.exists():
        paths.append(default)
    found, seen = [], set()
    for p in paths:
        try:
            with open(p, encoding="utf-8") as f:
                an = json.load(f)
        except Exception:
            continue
        city = an.get("city")
        if not city or city in seen:
            continue
        seen.add(city)
        found.append((city, an.get("n_units", 0), p))
    # richest case first
    return sorted(found, key=lambda x: -x[1])


for t in TASKS:
    t["count"] = len(ITEMS[t["id"]])
    t["status"] = "live" if t["count"] else "planned"
    # Task 3 has a bespoke change-detection view; any other task renders generically.
    t["view"] = "change" if t["id"] == "task3" else "generic"
    # a task shows an Evaluate tab if it's Task 3 or has a <id>_results.json
    t["has_eval"] = (t["id"] == "task3") or (DATA_DIR / f"{t['id']}_results.json").exists()
    # standalone supporting analysis (Task 3 only); not a benchmark task
    t["has_analysis"] = (t["id"] == "task3") and bool(analysis_files())

app = Flask(__name__)

def md_bold(text):
    """Render the **bold** markdown the model emits, as bold.

    The text is model output, so it is HTML-escaped first and only the bold
    spans are re-introduced as markup. Newlines become <br> so multi-line
    answers keep their structure.
    """
    from markupsafe import Markup, escape
    out = str(escape(text or ""))
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out, flags=re.S)
    out = out.replace("\n", "<br>")
    return Markup(out)

app.jinja_env.filters["mdbold"] = md_bold



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
    # land on Task 3 (the worked task with real results) if it's live
    landing = "task3" if TASK_BY_ID.get("task3", {}).get("status") == "live" else TASKS[0]["id"]
    return redirect(url_for("task_view", tid=landing))


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
    if not t or t["status"] != "live" or not t.get("has_eval"):
        abort(404)
    if t["view"] != "change":
        # generic per-task evaluation (Tasks 1, 2, 4, 5)
        with open(DATA_DIR / f"{tid}_results.json", encoding="utf-8") as f:
            res = json.load(f)
        by_id = {it["question_id"]: it for it in ITEMS[tid]}
        return render_template("evaluate_generic.html", result=res, items_by_id=by_id,
                               **nav(tid, "evaluate"))
    from src.baseline import majority_baseline
    from src.evaluate import evaluate_all
    items = ITEMS[tid]
    preds = majority_baseline(items)
    report = evaluate_all(items, preds)
    by_id = {it["question_id"]: it for it in items}
    # optional pre-computed scoreboard from src/scoreboard.py (data/results.json)
    scoreboard = None
    results_path = DATA_DIR / "results.json"
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            scoreboard = json.load(f)
    # the LLM row carries the real judge-based reasoning score
    llm_row = None
    if scoreboard:
        llm_row = next((r for r in scoreboard.get("scoreboard", [])
                        if r.get("reasoning") is not None), None)

    # Probe control: the same model on the same distributions, asked factual
    # questions that bypass the label schema. Shown here because its value is
    # entirely as a foil to the label accuracy above.
    probe = None
    p_res = DATA_DIR / "task3probe_results.json"
    p_items = DATA_DIR / "task3probe_items.jsonl"
    if p_res.exists() and p_items.exists():
        try:
            with open(p_res, encoding="utf-8") as f:
                pr = json.load(f)
            tier = {}
            with open(p_items, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        it = json.loads(line)
                        tier[it["question_id"]] = it["metadata"].get("tier", "other")
            tiers = {}
            for row in pr.get("rows", []):
                k = tier.get(row["question_id"], "other")
                d = tiers.setdefault(k, {"n": 0, "hit": 0})
                d["n"] += 1
                d["hit"] += int(row["gold"] == row["pred"])
            order = ["lookup", "change", "relational"]
            probe = {
                "accuracy": pr.get("accuracy"), "macro_f1": pr.get("macro_f1"),
                "n_items": pr.get("n_items"),
                "n_cities": len({q.rsplit("_", 1)[0] for q in tier}),
                "tiers": [{"tier": k, "n": tiers[k]["n"],
                           "acc": tiers[k]["hit"] / tiers[k]["n"]}
                          for k in order if k in tiers],
            }
        except Exception:
            probe = None
    return render_template("evaluate.html", report=report, preds=preds, items_by_id=by_id,
                           scoreboard=scoreboard, llm_row=llm_row, probe=probe,
                           models=model_rows(), **nav(tid, "evaluate"))


# ---------- JSON API ----------
@app.route("/task/<tid>/analysis")
def analysis_view(tid):
    t = TASK_BY_ID.get(tid)
    if not t or not t.get("has_analysis"):
        abort(404)
    files = analysis_files()
    want = request.args.get("city")
    path = next((p for c, _, p in files if c == want), files[0][2])
    with open(path, encoding="utf-8") as f:
        an = json.load(f)
    cities = [{"city": c, "n_units": n} for c, n, _ in files]
    cov = None
    cov_path = DATA_DIR / "hood_coverage.json"
    if cov_path.exists():
        try:
            with open(cov_path, encoding="utf-8") as f:
                rows = json.load(f).get("cities", [])
            cov = {"analysed": sum(1 for r in rows if r["status"] == "analysed"),
                   "sparse": sum(1 for r in rows if r["status"] == "sparse coverage"),
                   "none": sum(1 for r in rows if r["status"] == "no sub-units"),
                   "rows": rows}
        except Exception:
            cov = None
    an["units"] = sorted(an["units"], key=lambda u: -(u["n_2013"] + u["n_2018"]))
    total = sum(u["n_2013"] + u["n_2018"] for u in an["units"]) or 1
    for u in an["units"]:
        u["share"] = (u["n_2013"] + u["n_2018"]) / total
        u["agrees"] = u["label"] == an["city_label"]
        u["top_deltas"] = list(u["theme_deltas"].items())[:3]
    return render_template("analysis.html", an=an, cities=cities, cov=cov,
                           **nav(tid, "analysis"))


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