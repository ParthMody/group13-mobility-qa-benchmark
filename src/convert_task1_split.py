import csv
import json
import argparse
from collections import Counter

QUESTION = ("Given the user's recent check-ins in {city}, what broad POI "
            "category is the most likely next stop?")


def read_any(path):
    if path.lower().endswith(".csv"):
        with open(path, encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def options_of(rec):
    o = rec.get("options")
    if isinstance(o, list):
        return [str(x).strip() for x in o if str(x).strip()]
    return [p.strip() for p in str(o or "").split("|") if p.strip()]


def convert(rec):
    city = str(rec.get("city", "")).strip()
    choices = options_of(rec)
    answer = str(rec.get("correct_answer", "")).strip()
    atype = ("closed" if "multiple" in str(rec.get("answer_type", "")).lower()
             else "open")
    return {
        "question_id": str(rec.get("question_id", "")).strip(),
        "task": "task1_next_poi_category",
        "city": city,
        "user_id": "ANONYMISED",
        "context_sequence": [],
        "context_text": str(rec.get("context", "")).strip(),
        "target_time": "",
        "question": QUESTION.format(city=city),
        "choices": choices,
        "answer": answer,
        "rationale": str(rec.get("reasoning", "")).strip(),
        "source_dataset": str(rec.get("source_dataset", "massive_steps")).strip(),
        "metadata": {
            "answer_type": atype,
            "eval_mode": "classification",
            "difficulty": str(rec.get("difficulty", "")).strip(),
            "verification_status": str(rec.get("verification_status", "")).strip(),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="task1_initial_400.jsonl or .csv")
    ap.add_argument("--out", default="data/task1_items.jsonl")
    args = ap.parse_args()

    raw = read_any(args.source)
    items = [convert(r) for r in raw]

    problems = []
    seen = set()
    for it in items:
        q = it["question_id"]
        if not q:
            problems.append("record with no question_id")
        elif q in seen:
            problems.append(f"{q}: duplicate question_id")
        seen.add(q)
        if len(it["choices"]) < 2:
            problems.append(f"{q}: fewer than 2 options")
        if not it["answer"]:
            problems.append(f"{q}: no answer")
        elif it["answer"] not in it["choices"]:
            problems.append(f"{q}: answer not among options")
        if not it["context_text"]:
            problems.append(f"{q}: empty context")

    print(f"\n  read {len(raw)} records from {args.source}")
    print(f"  cities     : {len(Counter(i['city'] for i in items))}")
    print(f"  difficulty : {dict(Counter(i['metadata']['difficulty'] for i in items))}")
    print(f"  answers    : {len(Counter(i['answer'] for i in items))} distinct")

    if problems:
        print(f"\n  {len(problems)} problem(s) -- NOT writing:")
        for p in problems[:10]:
            print(f"    {p}")
        raise SystemExit(1)

    with open(args.out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"\n  wrote {len(items)} items -> {args.out}\n")


if __name__ == "__main__":
    main()