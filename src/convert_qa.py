import re
import json
import argparse

TASK_NAMES = {
    "task1": "task1_next_poi_category",
    "task2": "task2_weekday_weekend",
    "task4": "task4_zeroshot_reasoning",
    "task5": "task5_preference_shift",
}


def norm_key(k):
    k = k.strip().lower()
    if k.startswith("question"): return "question"
    if k.startswith("city"): return "city"
    if k.startswith("context"): return "context"
    if k.startswith("answer type"): return "answer_type"
    if k.startswith("options"): return "options"
    if k.startswith("correct") or k.startswith("reference answer"): return "answer"
    if k.startswith("reasoning"): return "rationale"
    if k.startswith("difficulty"): return "difficulty"
    if k.startswith("reference ranking"): return "reference_ranking"
    if k.startswith("marking guide"): return "scoring_rubric"
    return None


def parse_notion(text, task_id):
    items, cur = [], None
    for raw in text.splitlines():
        line = raw.strip()
        if re.match(r"^\*\*item\b", line, re.IGNORECASE):   # new item block
            if cur:
                items.append(cur)
            cur = {}
            continue
        if cur is None:
            continue
        m = re.match(r"^-\s*([^:]+):\s*(.*)$", line)
        if not m:
            continue
        key, val = norm_key(m.group(1)), m.group(2).strip()
        if not key:
            continue
        if key == "options":
            val = "" if "leave blank" in val.lower() else val
            cur["choices"] = [c.strip() for c in val.split("|") if c.strip()]
        elif key == "scoring_rubric":
            cur["scoring_rubric"] = [c.strip() for c in re.split(r"·|\|", val) if c.strip()]
        else:
            cur[key] = val
    if cur:
        items.append(cur)

    out = []
    n = 0
    for it in items:
        if not it.get("question"):   # skip the empty template block
            continue
        n += 1
        atype = "closed" if "multiple" in it.get("answer_type", "").lower() else "open"
        meta = {
            "answer_type": atype,
            "eval_mode": "classification" if atype == "closed" else "reasoning",
            "difficulty": it.get("difficulty", "").lower() or "medium",
        }
        if it.get("scoring_rubric"):
            meta["scoring_rubric"] = it["scoring_rubric"]
        if it.get("reference_ranking"):
            meta["reference_ranking"] = it["reference_ranking"]
        out.append({
            "question_id": f"{task_id}_{n:03d}",
            "task": TASK_NAMES[task_id],
            "city": it.get("city", ""),
            "user_id": "QA_AUTHORED",
            "context_sequence": [],
            "context_text": it.get("context", ""),
            "target_time": "",
            "question": it["question"],
            "choices": it.get("choices", []),
            "answer": it.get("answer", ""),
            "rationale": it.get("rationale", ""),
            "source_dataset": "massive_steps",
            "metadata": meta,
        })
    return out


def passthrough_jsonl(text):
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASK_NAMES))
    ap.add_argument("--notion")
    ap.add_argument("--jsonl")
    args = ap.parse_args()

    if args.notion:
        items = parse_notion(open(args.notion, encoding="utf-8").read(), args.task)
    elif args.jsonl:
        items = passthrough_jsonl(open(args.jsonl, encoding="utf-8").read())
    else:
        raise SystemExit("pass --notion FILE or --jsonl FILE")

    out_path = f"data/{args.task}_items.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{args.task}: wrote {len(items)} items -> {out_path}")
    if items:
        print("  first:", items[0].get("question", "")[:70])


if __name__ == "__main__":
    main()