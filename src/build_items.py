import json
import argparse
from collections import Counter, defaultdict

LABEL_SET = ["shift_to_transit", "shift_to_leisure", "shift_to_food",
             "increased_diversity", "no_major_change"]

# Crude category -> theme map. Extend with the real category taxonomy
# (semantic-trails categories.csv) for production items.
THEME = {
    "transit": ["train", "metro", "subway", "bus", "station", "platform", "tram", "airport"],
    "leisure": ["mall", "cinema", "park", "gym", "theater", "theatre", "shopping", "gallery", "museum"],
    "food":    ["cafe", "café", "coffee", "restaurant", "bar", "food", "bakery", "diner"],
}


def theme_of(cat):
    c = (cat or "").lower()
    for theme, kws in THEME.items():
        if any(k in c for k in kws):
            return theme
    return "other"


def period_of(trail_id):
    s = str(trail_id)
    if s.startswith("2013_"):
        return "A_2013"
    if s.startswith("2018_"):
        return "B_2018"
    return None


def category_mix(rows, top=8):
    c = Counter(r["venue_category"] for r in rows if r.get("venue_category"))
    total = sum(c.values()) or 1
    return {k: round(v / total, 3) for k, v in c.most_common(top)}


def decide_label(mix_a, mix_b):
    def theme_share(mix):
        ts = defaultdict(float)
        for cat, share in mix.items():
            ts[theme_of(cat)] += share
        return ts

    a, b = theme_share(mix_a), theme_share(mix_b)
    deltas = {t: b.get(t, 0) - a.get(t, 0) for t in set(a) | set(b)}
    if len(mix_b) - len(mix_a) >= 3:
        return "increased_diversity"
    if not deltas:
        return "no_major_change"
    top = max(deltas, key=lambda t: deltas[t])
    if deltas[top] < 0.05:
        return "no_major_change"
    return {"transit": "shift_to_transit", "leisure": "shift_to_leisure",
            "food": "shift_to_food"}.get(top, "no_major_change")


def reference_answer(city, unit, mix_a, mix_b, label):
    keys = set(mix_a) | set(mix_b)
    risers = sorted(((k, mix_b.get(k, 0) - mix_a.get(k, 0)) for k in keys), key=lambda x: -x[1])
    fallers = sorted(((k, mix_b.get(k, 0) - mix_a.get(k, 0)) for k in keys), key=lambda x: x[1])
    up = ", ".join(f"{k} ({d:+.0%})" for k, d in risers[:2] if d > 0)
    down = ", ".join(f"{k} ({d:+.0%})" for k, d in fallers[:2] if d < 0)
    subj = "this city" if unit == "city" else "this user"
    return (f"Between 2013 and 2018, {subj} ({city}) shows {label.replace('_', ' ')}. "
            f"Rising: {up or 'none notable'}. Declining: {down or 'none notable'}.")


def make_item(qid, city, user_id, mix_a, mix_b, label, unit):
    return {
        "question_id": qid, "task": "task3_change_detection", "city": city, "user_id": user_id,
        "context_sequence": [
            {"period": "A_2013", "category_mix": mix_a},
            {"period": "B_2018", "category_mix": mix_b},
        ],
        "target_time": "2013_vs_2018",
        "question": f"How did the visitation pattern for this {'city' if unit == 'city' else 'user'} "
                    f"change between 2013 and 2018?",
        "choices": [],
        "answer": reference_answer(city, unit, mix_a, mix_b, label),
        "rationale": "Change label derived from the largest theme-level shift between the two periods.",
        "source_dataset": "massive_steps",
        "metadata": {
            "answer_type": "open", "eval_mode": "both", "unit": unit,
            "change_label": label, "label_set": LABEL_SET,
            "scoring_rubric": [
                "identifies the main rising category or theme",
                "identifies at least one declining category",
                "does not invent categories absent from the data",
            ],
            "difficulty": "medium", "seed": False,
        },
    }


def load_rows(path):
    if path.endswith(".parquet"):
        import pandas as pd
        return pd.read_parquet(path).to_dict("records")
    if path.endswith(".csv"):
        import csv
        with open(path, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError("Provide a .parquet or .csv check-ins file")


def build(checkins_path, city, unit="city", min_per_period=5):
    rows = load_rows(checkins_path)
    total_read = len(rows)
    for r in rows:
        r["_period"] = period_of(r.get("trail_id"))
    rows = [r for r in rows if r["_period"]]
    n_a = sum(1 for r in rows if r["_period"] == "A_2013")
    n_b = sum(1 for r in rows if r["_period"] == "B_2018")
    print(f"  read {total_read} rows | with 2013/2018 prefix: {len(rows)} "
          f"(2013={n_a}, 2018={n_b})")
    items = []
    if unit == "city":
        a = [r for r in rows if r["_period"] == "A_2013"]
        b = [r for r in rows if r["_period"] == "B_2018"]
        mix_a, mix_b = category_mix(a), category_mix(b)
        if not mix_a or not mix_b:
            print(f"  SKIP {city}: empty period (2013 rows={len(a)}, 2018 rows={len(b)}) "
                  f"-> not a valid two-period item")
            return []
        items.append(make_item(f"task3_{city.lower().replace(' ', '')}_city",
                               city, "CITY_AGG", mix_a, mix_b,
                               decide_label(mix_a, mix_b), "city"))
    else:
        by_user = defaultdict(list)
        for r in rows:
            by_user[r["user_id"]].append(r)
        i = 0
        for uid, urows in by_user.items():
            a = [r for r in urows if r["_period"] == "A_2013"]
            b = [r for r in urows if r["_period"] == "B_2018"]
            if len(a) < min_per_period or len(b) < min_per_period:
                continue  # need coverage in both periods, or the change is noise
            mix_a, mix_b = category_mix(a), category_mix(b)
            i += 1
            items.append(make_item(f"task3_{city.lower().replace(' ', '')}_u{i:04d}",
                                   city, str(uid), mix_a, mix_b,
                                   decide_label(mix_a, mix_b), "user"))
    return items


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkins", required=True)
    ap.add_argument("--city", required=True)
    ap.add_argument("--unit", default="city", choices=["city", "user"])
    ap.add_argument("--out", default="data/task3_items.jsonl")
    args = ap.parse_args()

    print(f"Building {args.unit}-level items for {args.city} from {args.checkins}")
    built = build(args.checkins, args.city, args.unit)
    with open(args.out, "a", encoding="utf-8") as f:
        for it in built:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"Wrote {len(built)} item(s) to {args.out}\n")