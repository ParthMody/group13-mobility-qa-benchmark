import json
import os

NEEDED = ["question_id", "question", "choices", "answer"]


def scan(path):
    rows, bad = [], 0
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    bad += 1
    except Exception as e:
        return None, f"unreadable ({type(e).__name__})"
    return rows, (f"{bad} unparseable lines" if bad else "")


def main():
    found = []
    for root, _dirs, files in os.walk("data"):
        for fn in files:
            if fn.endswith(".jsonl") and "task1" in fn.lower():
                found.append(os.path.join(root, fn))

    if not found:
        print("No task1 .jsonl files found under data/")
        return

    print()
    for path in sorted(found):
        size = os.path.getsize(path)
        rows, note = scan(path)
        print(f"  {path}")
        print(f"    size: {size:,} bytes", f"  [{note}]" if note else "")

        if not rows:
            print("    -> EMPTY or unreadable. Not usable.\n")
            continue

        keys = set(rows[0].keys())
        missing = [k for k in NEEDED if k not in keys]
        cities = {}
        diffs = {}
        answered = 0
        valid_choice = 0
        for r in rows:
            cities[r.get("city", "?")] = cities.get(r.get("city", "?"), 0) + 1
            d = (r.get("metadata") or {}).get("difficulty", "?")
            diffs[d] = diffs.get(d, 0) + 1
            a = str(r.get("answer", "")).strip()
            if a:
                answered += 1
                if a in (r.get("choices") or []):
                    valid_choice += 1

        print(f"    items: {len(rows)}")
        print(f"    cities: {len(cities)}  ->  "
              + ", ".join(f"{k}({v})" for k, v in
                          sorted(cities.items(), key=lambda x: -x[1])[:6]))
        print(f"    difficulty: " + ", ".join(f"{k}={v}" for k, v in sorted(diffs.items())))
        print(f"    has answer: {answered}/{len(rows)}"
              f"   answer in choices: {valid_choice}/{len(rows)}")
        if missing:
            print(f"    MISSING required fields: {missing}")
        else:
            print(f"    schema: ok for eval_task.py")
        print()

    print("  The evaluator reads data/task1_items.jsonl.")
    print("  Copy whichever file above is the intended set to that path before running.\n")


if __name__ == "__main__":
    main()