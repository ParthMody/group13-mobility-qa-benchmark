import os
import json
from collections import Counter

ITEMS = "data/task3_items.jsonl"
SOURCES = [("Majority baseline", None),
           ("ML (gradient boosting)", "data/preds_ml.jsonl"),
           ("LLM zero-shot", "data/preds_llm.jsonl"),
           ("LLM zero-shot + CoT", "data/preds_llm_cot.jsonl")]


def load_jsonl(path):
    out = {}
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    out[r["question_id"]] = r
    return out


def macro_f1(pairs, labels):
    f1s = []
    for lab in labels:
        tp = sum(1 for g, p in pairs if g == lab and p == lab)
        fp = sum(1 for g, p in pairs if g != lab and p == lab)
        fn = sum(1 for g, p in pairs if g == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def main():
    items = [json.loads(l) for l in open(ITEMS, encoding="utf-8") if l.strip()]
    gold = {it["question_id"]: it["metadata"]["change_label"] for it in items}
    labels = items[0]["metadata"]["label_set"]
    majority = Counter(gold.values()).most_common(1)[0][0]

    rows = []
    for name, path in SOURCES:
        if path is None:
            pred = {qid: majority for qid in gold}
            reasoning = None
        else:
            data = load_jsonl(path)
            if not data:
                continue
            pred = {qid: data.get(qid, {}).get("change_label", "invalid") for qid in gold}
            rscores = [data[q]["reasoning_score"] for q in data if "reasoning_score" in data[q]]
            reasoning = round(sum(rscores) / len(rscores), 3) if rscores else None
        pairs = [(gold[q], pred[q]) for q in gold]
        acc = sum(1 for g, p in pairs if g == p) / len(pairs)
        rows.append({"method": name, "accuracy": round(acc, 3),
                     "macro_f1": round(macro_f1(pairs, labels), 3),
                     "reasoning": reasoning})

    result = {"n_items": len(items), "labels": labels, "scoreboard": rows}
    with open("data/results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== Task 3 scoreboard ({len(items)} items) ===")
    print(f"{'method':26} {'acc':>6} {'macroF1':>8} {'reasoning':>10}")
    for r in rows:
        print(f"{r['method']:26} {r['accuracy']:>6} {r['macro_f1']:>8} "
              f"{str(r['reasoning']) if r['reasoning'] is not None else '-':>10}")
    print("\nWrote data/results.json")


if __name__ == "__main__":
    main()