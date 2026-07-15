import os
import re
import json
import time
import argparse

ITEMS_DEFAULT = "data/task3_items.jsonl"


# ---------- provider callers ----------
def get_caller(provider):
    """Return call(model, prompt) -> str, for the chosen provider."""
    if provider == "gemini":
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise SystemExit("GEMINI_API_KEY not set. Run:  set GEMINI_API_KEY=your-key")
        try:  # new SDK: pip install google-genai
            from google import genai
            client = genai.Client(api_key=key)

            def call(model, prompt):
                return client.models.generate_content(model=model, contents=prompt).text
            return call
        except ImportError:  # legacy SDK: pip install google-generativeai
            import google.generativeai as genai
            genai.configure(api_key=key)

            def call(model, prompt):
                return genai.GenerativeModel(model).generate_content(prompt).text
            return call

    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY not set.")
        from anthropic import Anthropic
        client = Anthropic()

        def call(model, prompt):
            r = client.messages.create(model=model, max_tokens=400,
                                       messages=[{"role": "user", "content": prompt}])
            return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        return call

    raise SystemExit(f"Unknown provider: {provider}")


def with_retry(call, model, prompt, retries=3, base_sleep=5):
    for attempt in range(retries + 1):
        try:
            return call(model, prompt)
        except Exception as e:
            if attempt == retries:
                raise
            wait = base_sleep * (attempt + 1)
            print(f"  ...call failed ({type(e).__name__}); retry in {wait}s")
            time.sleep(wait)


# ---------- items / prompt / parsing ----------
def load_items(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def mixes(item):
    seq = item["context_sequence"]
    return seq[0].get("category_mix", {}), seq[1].get("category_mix", {})


def fmt_mix(m):
    return ", ".join(f"{k}: {round(v*100)}%" for k, v in m.items()) or "(none)"


def build_prompt(item, cot=False):
    a, b = mixes(item)
    labels = item["metadata"]["label_set"]
    think = "First reason briefly about which theme moved most, then answer. " if cot else ""
    return (
        f"You are analysing how a {item['metadata']['unit']}'s point-of-interest "
        f"visitation pattern changed between 2013 and 2018 in {item['city']}.\n\n"
        f"2013 category mix: {fmt_mix(a)}\n"
        f"2018 category mix: {fmt_mix(b)}\n\n"
        f"Choose the single best change label from this exact set:\n{labels}\n\n"
        f"{think}Respond with ONLY a JSON object, no other text:\n"
        f'{{"label": "<one label from the set>", "explanation": "<one or two sentences>"}}'
    )


def parse_json(text):
    text = re.sub(r"^```(json)?|```$", "", (text or "").strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


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


def judge_reasoning(call, model, item, explanation, mock=False):
    rubric = item["metadata"].get("scoring_rubric", [])
    if mock:
        hits = sum(1 for c in rubric if any(w in explanation.lower()
                   for w in c.lower().split() if len(w) > 4))
        return round(min(1.0, hits / max(1, len(rubric))), 2)
    prompt = ("Score the ANSWER against the rubric. Return ONLY JSON "
              '{"score": <0..1>, "notes": "<short>"}.\n\n'
              "RUBRIC:\n- " + "\n- ".join(rubric) + f"\n\nANSWER:\n{explanation}\n")
    parsed = parse_json(with_retry(call, model, prompt)) or {}
    try:
        return max(0.0, min(1.0, float(parsed.get("score", 0))))
    except Exception:
        return 0.0


def mock_reply(item):
    a, b = mixes(item)
    gold = item["metadata"]["change_label"]
    return json.dumps({"label": gold,
                       "explanation": f"Mock: mix shifts from [{fmt_mix(a)}] to "
                                      f"[{fmt_mix(b)}], consistent with {gold.replace('_',' ')}."})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=ITEMS_DEFAULT)
    ap.add_argument("--provider", default="gemini", choices=["gemini", "anthropic"])
    ap.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
    ap.add_argument("--cot", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--sleep", type=float, default=4.0, help="seconds between calls (free-tier rate limit)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    items = load_items(args.items)
    labels = items[0]["metadata"]["label_set"]
    call = None if args.mock else get_caller(args.provider)

    gold, pred, preds_out, rsum = [], [], [], 0.0
    for i, it in enumerate(items, 1):
        prompt = build_prompt(it, cot=args.cot)
        raw = mock_reply(it) if args.mock else with_retry(call, args.model, prompt)
        obj = parse_json(raw) or {}
        label = obj.get("label", "").strip()
        if label not in labels:
            label = "invalid"
        expl = obj.get("explanation", "")
        if not args.mock and args.sleep:
            time.sleep(args.sleep)
        r = judge_reasoning(call, args.model, it, expl, mock=args.mock)
        rsum += r
        gold.append(it["metadata"]["change_label"]); pred.append(label)
        preds_out.append({"question_id": it["question_id"], "change_label": label,
                          "explanation": expl, "reasoning_score": r})
        print(f"  [{i}/{len(items)}] {it['city']:12} gold={gold[-1]:20} pred={label}")
        if not args.mock and args.sleep:
            time.sleep(args.sleep)

    acc = sum(1 for g, p in zip(gold, pred) if g == p) / len(gold)
    f1 = macro_f1(gold, pred, labels)
    out = args.out or ("data/preds_llm_cot.jsonl" if args.cot else "data/preds_llm.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for p in preds_out:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    tag = f"{args.provider} zero-shot" + (" + CoT" if args.cot else "") + (" [MOCK]" if args.mock else "")
    print(f"\n=== {tag} ({args.model}) ===")
    print(f"  items          : {len(items)}")
    print(f"  label accuracy : {acc:.3f}")
    print(f"  macro-F1       : {f1:.3f}")
    print(f"  mean reasoning : {rsum/len(items):.3f}")
    print(f"  wrote          : {out}")


if __name__ == "__main__":
    main()