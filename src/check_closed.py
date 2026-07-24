import json
import math
import argparse
from collections import Counter, defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", help="task1, task2, task5, ...")
    args = ap.parse_args()

    items = {}
    with open(f"data/{args.task}_items.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                items[d["question_id"]] = d

    with open(f"data/{args.task}_results.json", encoding="utf-8") as f:
        res = json.load(f)
    rows = res["rows"]
    n = len(rows)

    if res.get("type") == "open" or res.get("accuracy") is None:
        raise SystemExit(
            f"{args.task} is an open (reasoning-scored) task, not a closed "
            f"multiple-choice task. Use:  python -m src.check_open {args.task}"
        )

    ks = [len(items[r["question_id"]]["choices"]) for r in rows
          if r["question_id"] in items and items[r["question_id"]]["choices"]]
    if not ks:
        raise SystemExit(
            f"{args.task}: no items with choices matched the result rows — "
            f"nothing to score as multiple-choice."
        )
    chance = sum(1.0 / k for k in ks) / len(ks)
    acc = res["accuracy"]
    se = math.sqrt(chance * (1 - chance) / n)
    z = (acc - chance) / se

    print(f"\n=== {args.task}: {n} items ===\n")
    print(f"  accuracy          : {acc:.3f}")
    print(f"  macro-F1          : {res['macro_f1']:.3f}")
    print(f"  chance (mean 1/k) : {chance:.3f}   options per item: {dict(Counter(ks))}")
    print(f"  standard error    : {se:.3f}")
    print(f"  z vs chance       : {z:+.2f}")
    if abs(z) < 1.96:
        print(f"  -> INDISTINGUISHABLE from chance")
    else:
        direction = "ABOVE" if z > 0 else "BELOW"
        print(f"  -> significantly {direction} chance (p < 0.05)")

    golds = Counter(r["gold"] for r in rows)
    preds = Counter(r["pred"] for r in rows)
    top_gold, top_n = golds.most_common(1)[0]
    print(f"  majority baseline : {top_n/n:.3f}  (always predict {top_gold})")

    print(f"\n  predicted vs gold")
    print(f"    {'class':<30}{'pred':>7}{'gold':>7}{'ratio':>9}")
    print("    " + "-" * 53)
    for c in sorted(set(golds) | set(preds), key=lambda c: -preds.get(c, 0)):
        p, g = preds.get(c, 0), golds.get(c, 0)
        print(f"    {c:<30}{p:>7}{g:>7}{(f'{p/g:.2f}x' if g else '--'):>9}")

    hit = Counter(r["gold"] for r in rows if r["gold"] == r["pred"])
    print(f"\n  recall per class")
    for c, g in sorted(golds.items(), key=lambda x: hit.get(x[0], 0) / x[1]):
        print(f"    {c:<30}{hit.get(c,0):>5}/{g:<5}{hit.get(c,0)/g:>7.0%}")

    for key, label in [("difficulty", "difficulty"), ("city", "city")]:
        groups = defaultdict(lambda: [0, 0])
        for r in rows:
            it = items.get(r["question_id"])
            if not it:
                continue
            v = (it["metadata"].get(key) if key == "difficulty"
                 else it.get(key, "?")) or "?"
            groups[v][1] += 1
            if r["gold"] == r["pred"]:
                groups[v][0] += 1
        if len(groups) > 1:
            print(f"\n  by {label}")
            order = (["easy", "medium", "hard"] if key == "difficulty"
                     else sorted(groups, key=lambda c: groups[c][0] / groups[c][1]))
            for v in order:
                if v in groups:
                    h, t = groups[v]
                    zz = (h/t - chance) / math.sqrt(chance*(1-chance)/t)
                    print(f"    {v:<16}{h:>4}/{t:<5}{h/t:>7.3f}   z={zz:+.2f}")
    print()


if __name__ == "__main__":
    main()