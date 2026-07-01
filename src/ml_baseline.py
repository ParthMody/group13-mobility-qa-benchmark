import json
import math
import argparse
from collections import Counter

MIN_ITEMS = 30      # below this, don't pretend to train
MIN_PER_CLASS = 3


def load_items(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def entropy(mix):
    return -sum(v * math.log(v + 1e-9) for v in mix.values() if v > 0)


def featurize(item):
    a = item["context_sequence"][0].get("category_mix", {})
    b = item["context_sequence"][1].get("category_mix", {})
    keys = set(a) | set(b)
    new = [k for k in b if k not in a]
    dropped = [k for k in a if k not in b]
    top_a = max(a.values()) if a else 0.0
    top_b = max(b.values()) if b else 0.0
    return {
        "n_cat_a": len(a), "n_cat_b": len(b), "n_cat_delta": len(b) - len(a),
        "n_new": len(new), "n_dropped": len(dropped),
        "entropy_a": entropy(a), "entropy_b": entropy(b),
        "entropy_delta": entropy(b) - entropy(a),
        "top_share_a": top_a, "top_share_b": top_b,
        "top_share_delta": top_b - top_a,
        "overlap": len(keys & set(b) & set(a)) / (len(keys) or 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/task3_items.jsonl")
    ap.add_argument("--out", default="data/preds_ml.jsonl")
    args = ap.parse_args()

    items = load_items(args.items)
    labels = [it["metadata"]["change_label"] for it in items]
    dist = Counter(labels)
    classes = [c for c, n in dist.items() if n >= MIN_PER_CLASS]

    print(f"items: {len(items)}  |  label distribution: {dict(dist)}")
    if len(items) < MIN_ITEMS or len(classes) < 2:
        print(f"\nInsufficient data for a real ML baseline "
              f"(need >= {MIN_ITEMS} items and >= 2 classes with >= {MIN_PER_CLASS} each).")
        print("This is the expected outcome on the seed set / a small per-user overlap.")
        print("-> Report the ZERO-SHOT LLM result as the headline; note ML as data-limited.")
        return

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

    X = DictVectorizer(sparse=False).fit_transform([featurize(it) for it in items])
    y = labels
    majority = dist.most_common(1)[0][0]
    maj_acc = sum(1 for t in y if t == majority) / len(y)
    print(f"majority baseline acc: {maj_acc:.3f} (always '{majority}')")

    cv = StratifiedKFold(n_splits=min(5, min(dist.values())), shuffle=True, random_state=0)
    for name, clf in [("logreg", LogisticRegression(max_iter=1000)),
                      ("gboost", GradientBoostingClassifier(random_state=0))]:
        pred = cross_val_predict(clf, X, y, cv=cv)
        print(f"\n=== {name} (cross-validated) ===")
        print(f"  accuracy : {accuracy_score(y, pred):.3f}")
        print(f"  macro-F1 : {f1_score(y, pred, average='macro'):.3f}")
        print(classification_report(y, pred, zero_division=0))
        if name == "gboost":
            with open(args.out, "w", encoding="utf-8") as f:
                for it, p in zip(items, pred):
                    f.write(json.dumps({"question_id": it["question_id"],
                                        "change_label": p}, ensure_ascii=False) + "\n")
            print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()