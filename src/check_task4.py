import json
from collections import defaultdict

ITEMS = "data/task4_items.jsonl"
RESULTS = "data/task4_results.json"


def main():
    items = {}
    with open(ITEMS, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                items[d["question_id"]] = d

    with open(RESULTS, encoding="utf-8") as f:
        res = json.load(f)
    rows = res["rows"]

    # ---- 1. tier breakdown ----
    tiers = defaultdict(list)
    for r in rows:
        it = items.get(r["question_id"])
        if it:
            tiers[it["metadata"]["difficulty"]].append(r.get("reasoning_score", 0.0))

    print(f"\n=== Task 4: {len(rows)} items, reported mean {res.get('reasoning')} ===\n")
    print("  tier breakdown")
    print(f"    {'tier':<8}{'n':>5}{'mean':>8}")
    print("    " + "-" * 21)
    for tier in ["easy", "medium", "hard"]:
        v = tiers.get(tier)
        if v:
            print(f"    {tier:<8}{len(v):>5}{sum(v)/len(v):>8.3f}")

    # ---- 2. judge failures ----
    exact, partial = [], []
    for r in rows:
        it = items.get(r["question_id"])
        if not it:
            continue
        gold = it["answer"].strip()
        said = r.get("answer", "")
        score = r.get("reasoning_score", 0.0)
        if score > 0.5:
            continue
        if gold.lower() in said.lower():
            (exact if score == 0.0 else partial).append((r["question_id"], gold, score))

    n = len(rows)
    total = len(exact) + len(partial)
    print(f"\n  judge failures (answer names the gold category, score <= 0.5)")
    print(f"    scored 0.0      : {len(exact):>4} / {n}  ({len(exact)/n:.0%})")
    print(f"    scored 0.1-0.5  : {len(partial):>4} / {n}  ({len(partial)/n:.0%})")
    print(f"    total           : {total:>4} / {n}  ({total/n:.0%})")

    if exact:
        print(f"\n    examples scored 0.0 despite naming the gold category:")
        for q, gold, s in exact[:12]:
            print(f"      {q.replace('task4_cat_', ''):<34} gold={gold}")

    if total:
        floor = res.get("reasoning", 0)
        print(f"\n  The reported mean of {floor} is a floor: at least {total} items")
        print(f"  ({total/n:.0%}) were scored low despite naming the correct category.")


if __name__ == "__main__":
    main()