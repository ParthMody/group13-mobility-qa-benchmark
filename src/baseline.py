from collections import Counter


def majority_baseline(items):
    """Predict the single most common change_label for every item."""
    labels = [it.get("metadata", {}).get("change_label") for it in items
              if it.get("metadata", {}).get("change_label")]
    majority = Counter(labels).most_common(1)[0][0] if labels else None
    preds = {}
    for it in items:
        preds[it["question_id"]] = {
            "change_label": majority,
            "answer": (f"The pattern appears to show {majority.replace('_', ' ')}."
                       if majority else ""),
        }
    return preds