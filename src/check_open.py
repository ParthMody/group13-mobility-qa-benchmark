import json
import math
import argparse
from collections import Counter, defaultdict


def summary(scores):
    n = len(scores)
    mean = sum(scores) / n
    var = sum((s - mean) ** 2 for s in scores) / n
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    return mean, sd, se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", help="task4, ... (open reasoning-scored tasks)")
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

    scored = [r for r in rows if r.get("reasoning_score") is not None]
    if not scored:
        raise SystemExit(
            f"{args.task}: no rows carry a reasoning_score. If this is a "
            f"closed task, use:  python -m src.check_closed {args.task}"
        )
    scores = [r["reasoning_score"] for r in scored]
    n = len(scores)
    mean, sd, se = summary(scores)

    print(f"\n=== {args.task}: {n} items (open reasoning) ===\n")
    print(f"  model             : {res.get('model', '?')}")
    print(f"  mean reasoning    : {mean:.3f}")
    print(f"  std dev           : {sd:.3f}")
    print(f"  standard error    : {se:.3f}   95% CI [{mean-1.96*se:.3f}, {mean+1.96*se:.3f}]")

    dist = Counter(scores)
    print(f"\n  score distribution")
    for s in sorted(dist):
        c = dist[s]
        bar = "#" * round(40 * c / n)
        print(f"    {s:>4.2f}  {c:>4}  {c/n:>5.1%}  {bar}")

    # bucketed view: fail / partial / pass
    buckets = {"fail (<0.5)": 0, "partial [0.5,0.9)": 0, "pass (>=0.9)": 0}
    for s in scores:
        if s < 0.5:
            buckets["fail (<0.5)"] += 1
        elif s < 0.9:
            buckets["partial [0.5,0.9)"] += 1
        else:
            buckets["pass (>=0.9)"] += 1
    print(f"\n  buckets")
    for k, c in buckets.items():
        print(f"    {k:<20}{c:>5}  {c/n:>6.1%}")

    for key, label in [("difficulty", "difficulty"), ("city", "city")]:
        groups = defaultdict(list)
        for r in scored:
            it = items.get(r["question_id"])
            if not it:
                continue
            v = (it.get("metadata", {}).get(key) if key == "difficulty"
                 else it.get(key, "?")) or "?"
            groups[v].append(r["reasoning_score"])
        if len(groups) > 1:
            print(f"\n  by {label}")
            order = (["easy", "medium", "hard"] if key == "difficulty"
                     else sorted(groups, key=lambda c: sum(groups[c]) / len(groups[c])))
            for v in order:
                if v in groups:
                    g = groups[v]
                    m, _, e = summary(g)
                    print(f"    {v:<16}{len(g):>5}   mean={m:.3f}  se={e:.3f}")
    print()


if __name__ == "__main__":
    main()
