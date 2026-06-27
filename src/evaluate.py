"""
Evaluation for Task 3 (Two-Period Change Detection).

Two modes, matching the dual nature of the task:
  - classification: exact match on metadata.change_label (for the ML head)
  - open:           how well the written answer covers the scoring rubric

The open scorer here is a transparent placeholder (rubric keyword coverage) so
the harness runs end-to-end today. Swap in llm_judge() for real scoring later.

Prediction format (matches the shared harness):
    {question_id: {"change_label": "<label>", "answer": "<text>"}}
"""
from collections import defaultdict


def score_classification(items, predictions):
    total = correct = 0
    per_label = defaultdict(lambda: {"n": 0, "correct": 0})
    for it in items:
        gold = it.get("metadata", {}).get("change_label")
        if gold is None:
            continue
        total += 1
        pred = predictions.get(it["question_id"], {})
        pred_label = pred.get("change_label") if isinstance(pred, dict) else pred
        per_label[gold]["n"] += 1
        if pred_label == gold:
            correct += 1
            per_label[gold]["correct"] += 1
    acc = correct / total if total else 0.0
    return {
        "n": total,
        "accuracy": round(acc, 3),
        "per_label": {
            k: {"n": v["n"], "recall": round(v["correct"] / v["n"], 3) if v["n"] else 0.0}
            for k, v in per_label.items()
        },
    }


def _rubric_coverage(answer_text, rubric):
    """Placeholder: a rubric criterion counts as 'met' if a third of its salient
    words appear in the answer. Crude but transparent; replace with llm_judge()."""
    if not rubric:
        return None
    text = (answer_text or "").lower()
    met = 0
    for crit in rubric:
        words = [w for w in crit.lower().replace(",", " ").split() if len(w) > 4]
        hits = sum(1 for w in words if w in text)
        if words and hits >= max(1, len(words) // 3):
            met += 1
    return met / len(rubric)


def score_open(items, predictions):
    scores, detail = [], {}
    for it in items:
        md = it.get("metadata", {})
        if md.get("answer_type") != "open":
            continue
        pred = predictions.get(it["question_id"], {})
        ans = pred.get("answer") if isinstance(pred, dict) else pred
        cov = _rubric_coverage(ans, md.get("scoring_rubric", []))
        if cov is not None:
            scores.append(cov)
            detail[it["question_id"]] = round(cov, 3)
    mean = sum(scores) / len(scores) if scores else 0.0
    return {
        "n": len(scores),
        "mean_rubric_coverage": round(mean, 3),
        "per_item": detail,
        "note": "Heuristic placeholder. Wire up llm_judge() for real open-answer scoring.",
    }


def llm_judge(prediction, item):
    """Hook for real open-answer scoring via an LLM judge (e.g. Anthropic API),
    grading the prediction against item['metadata']['scoring_rubric']."""
    raise NotImplementedError("Plug an LLM judge in here to score open answers against the rubric.")


def evaluate_all(items, predictions):
    return {
        "classification": score_classification(items, predictions),
        "open": score_open(items, predictions),
        "n_items": len(items),
    }