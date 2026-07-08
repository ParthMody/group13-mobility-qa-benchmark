import os
import json
import time
import argparse

# reuse the provider caller, JSON parser, judge, and retry from llm_zeroshot.py
from src.llm_zeroshot import get_caller, with_retry, parse_json, judge_reasoning


def load_items(task):
    path = f"data/{task}_items.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def macro_f1(gold, pred, labels):
    f1s = []
    for lab in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == lab and p == lab)
        fp = sum(1 for g, p in zip(gold, pred) if g != lab and p == lab)
        fn = sum(1 for g, p in zip(gold, pred) if g == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def mcq_prompt(it):
    ctx = it.get("context_text", "")
    opts = "\n".join(f"- {c}" for c in it["choices"])
    return (f"{it['question']}\n\n"
            f"{('Context: ' + ctx + chr(10) + chr(10)) if ctx else ''}"
            f"Options:\n{opts}\n\n"
            f"Respond with ONLY the exact text of the single best option.")


def open_prompt(it):
    ctx = it.get("context_text", "")
    return (f"{it['question']}\n\n"
            f"{('Context: ' + ctx + chr(10) + chr(10)) if ctx else ''}"
            f"Answer in two or three sentences.")


def match_choice(reply, choices):
    reply = (reply or "").strip().lower()
    for c in choices:
        if c.lower() in reply or reply in c.lower():
            return c
    return "invalid"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--provider", default="gemini", choices=["gemini", "anthropic"])
    ap.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--sleep", type=float, default=4.0)
    args = ap.parse_args()

    items = load_items(args.task)
    call = None if args.mock else get_caller(args.provider)
    is_closed = (items[0].get("metadata", {}).get("answer_type") == "closed"
                 or bool(items[0].get("choices")))

    gold, pred, rows, rsum = [], [], [], 0.0
    for i, it in enumerate(items, 1):
        if is_closed:
            prompt = mcq_prompt(it)
            raw = it["answer"] if args.mock else with_retry(call, args.model, prompt)
            choice = match_choice(raw, it["choices"])
            gold.append(it["answer"]); pred.append(choice)
            rows.append({"question_id": it["question_id"], "gold": it["answer"], "pred": choice})
            print(f"  [{i}/{len(items)}] gold={it['answer'][:22]:22} pred={choice[:22]}")
        else:
            prompt = open_prompt(it)
            raw = "Mock answer." if args.mock else with_retry(call, args.model, prompt)
            if not args.mock and args.sleep:
                time.sleep(args.sleep)
            r = judge_reasoning(call, args.model, it, raw, mock=args.mock)
            rsum += r
            rows.append({"question_id": it["question_id"], "answer": raw, "reasoning_score": r})
            print(f"  [{i}/{len(items)}] reasoning={r}")
        if not args.mock and args.sleep:
            time.sleep(args.sleep)

    if is_closed:
        acc = sum(1 for g, p in zip(gold, pred) if g == p) / len(gold)
        labels = sorted(set(gold))
        result = {"task": args.task, "n_items": len(items), "type": "closed",
                  "accuracy": round(acc, 3), "macro_f1": round(macro_f1(gold, pred, labels), 3),
                  "reasoning": None, "rows": rows}
        print(f"\n{args.task}: accuracy={acc:.3f}  macro-F1={result['macro_f1']:.3f}")
    else:
        result = {"task": args.task, "n_items": len(items), "type": "open",
                  "accuracy": None, "macro_f1": None,
                  "reasoning": round(rsum / len(items), 3), "rows": rows}
        print(f"\n{args.task}: mean reasoning={result['reasoning']:.3f}")

    out = f"data/{args.task}_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()