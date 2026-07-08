import json
import argparse
from collections import Counter, defaultdict

LABEL_SET = ["shift_to_transit", "shift_to_food", "shift_to_leisure",
             "shift_to_shopping", "shift_to_outdoors", "shift_to_residential",
             "shift_to_work", "increased_diversity", "no_major_change"]

# Category -> theme taxonomy. Keys are checked in order (first match wins), so
# more specific themes (food, leisure) come before broad ones (shopping) to keep
# e.g. "Coffee Shop" in food rather than shopping.
THEME = {
    "home":     ["home", "residential", "apartment", "condo", "housing", "neighborhood",
                 "neighbourhood", "dorm", "house"],
    "work":     ["office", "coworking", "corporate", "cubicle", "conference room", "workplace"],
    "transit":  ["train", "metro", "subway", "bus", "station", "platform", "tram", "light rail",
                 "airport", "terminal", "gate", "ferry", "boat", "road", "bridge", "transport",
                 "taxi", "parking", "highway"],
    "food":     ["cafe", "café", "coffee", "restaurant", "bar", "pub", "food", "bakery", "diner",
                 "deli", "pizza", "noodle", "ramen", "soba", "sushi", "bistro", "brewery",
                 "winery", "tea room", "dessert", "eatery", "steakhouse", "grill", "snack"],
    "leisure":  ["cinema", "theater", "theatre", "gallery", "museum", "gym", "fitness", "nightclub",
                 "club", "stadium", "arena", "entertainment", "arcade", "bowling", "spa", "concert"],
    "outdoors": ["park", "beach", "waterfront", "trail", "garden", "mountain", "outdoors", "lake",
                 "river", "pier", "monument", "landmark", "harbor", "harbour", "plaza", "scenic"],
    "education":["university", "college", "school", "campus", "library", "classroom", "lab", "student"],
    "civic":    ["church", "temple", "mosque", "synagogue", "shrine", "hospital", "medical", "clinic",
                 "government", "court", "police", "embassy", "community", "civic", "county", "city"],
    "shopping": ["mall", "shop", "store", "market", "boutique", "supermarket", "grocery", "retail",
                 "outlet", "pharmacy", "bookstore", "department"],
}

# themes we can name as a destination in a "shift_to_X" label
THEME_LABEL = {"transit": "shift_to_transit", "food": "shift_to_food",
               "leisure": "shift_to_leisure", "shopping": "shift_to_shopping",
               "outdoors": "shift_to_outdoors", "home": "shift_to_residential",
               "work": "shift_to_work"}


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
    """Top-N category shares, for DISPLAY in the item (not used for labelling)."""
    c = Counter(r["venue_category"] for r in rows if r.get("venue_category"))
    total = sum(c.values()) or 1
    return {k: round(v / total, 3) for k, v in c.most_common(top)}


def theme_shares(rows):
    """Theme shares over the FULL distribution of a period, used for labelling."""
    c = Counter(theme_of(r.get("venue_category")) for r in rows if r.get("venue_category"))
    total = sum(c.values()) or 1
    return {t: v / total for t, v in c.items()}


def _entropy(dist):
    import math
    return -sum(p * math.log(p + 1e-9) for p in dist.values() if p > 0)


def decide_label(theme_a, theme_b, thresh=0.05, div_thresh=0.20):
    """Label by the dominant theme the visits shifted TOWARD, over full theme
    shares. Falls back to increased_diversity (spread rose, no single winner) or
    no_major_change (nothing moved beyond `thresh`)."""
    keys = set(theme_a) | set(theme_b)
    deltas = {t: theme_b.get(t, 0) - theme_a.get(t, 0) for t in keys}
    risers = {t: d for t, d in deltas.items() if t in THEME_LABEL and d > 0}
    top_theme, top_delta = (max(risers.items(), key=lambda x: x[1]) if risers else (None, 0.0))
    ent_gain = _entropy(theme_b) - _entropy(theme_a)

    if top_delta < thresh:
        return "increased_diversity" if ent_gain > div_thresh else "no_major_change"
    # a broad, spread-out rise with no strong single winner reads as diversification
    if ent_gain > div_thresh and top_delta < 2 * thresh:
        return "increased_diversity"
    return THEME_LABEL[top_theme]


def reference_answer(city, unit, mix_a, mix_b, label, theme_a, theme_b):
    keys = set(mix_a) | set(mix_b)
    risers = sorted(((k, mix_b.get(k, 0) - mix_a.get(k, 0)) for k in keys), key=lambda x: -x[1])
    fallers = sorted(((k, mix_b.get(k, 0) - mix_a.get(k, 0)) for k in keys), key=lambda x: x[1])
    up = ", ".join(f"{k} ({d:+.0%})" for k, d in risers[:2] if d > 0)
    down = ", ".join(f"{k} ({d:+.0%})" for k, d in fallers[:2] if d < 0)
    # dominant theme movement, for context
    tkeys = set(theme_a) | set(theme_b)
    tdelta = sorted(((t, theme_b.get(t, 0) - theme_a.get(t, 0)) for t in tkeys), key=lambda x: -abs(x[1]))
    theme_note = ", ".join(f"{t} ({d:+.0%})" for t, d in tdelta[:2] if abs(d) >= 0.02)
    subj = "this city" if unit == "city" else "this user"
    return (f"Between 2013 and 2018, {subj} ({city}) shows {label.replace('_', ' ')}. "
            f"At the theme level the largest movement is in {theme_note or 'no theme notably'}. "
            f"Rising categories: {up or 'none notable'}. Declining: {down or 'none notable'}.")


def make_item(qid, city, user_id, mix_a, mix_b, label, unit, theme_a, theme_b):
    tkeys = set(theme_a) | set(theme_b)
    theme_deltas = {t: round(theme_b.get(t, 0) - theme_a.get(t, 0), 3) for t in tkeys}
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
        "answer": reference_answer(city, unit, mix_a, mix_b, label, theme_a, theme_b),
        "rationale": ("Change label is the theme (transit, food, leisure, shopping, outdoors, "
                      "residential, work) whose share of visits grew most from 2013 to 2018, "
                      "computed over the full check-in distribution; 'increased_diversity' when "
                      "the spread widens with no single dominant theme, and 'no_major_change' "
                      "when no theme moves beyond 5 percentage points."),
        "source_dataset": "massive_steps",
        "metadata": {
            "answer_type": "open", "eval_mode": "both", "unit": unit,
            "change_label": label, "label_set": LABEL_SET,
            "theme_shares_2013": {t: round(s, 3) for t, s in sorted(theme_a.items(), key=lambda x: -x[1])},
            "theme_shares_2018": {t: round(s, 3) for t, s in sorted(theme_b.items(), key=lambda x: -x[1])},
            "theme_deltas": dict(sorted(theme_deltas.items(), key=lambda x: -abs(x[1]))),
            "scoring_rubric": [
                "identifies the main rising theme or category",
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
        theme_a, theme_b = theme_shares(a), theme_shares(b)
        label = decide_label(theme_a, theme_b)
        print(f"  {city}: label={label}")
        items.append(make_item(f"task3_{city.lower().replace(' ', '')}_city",
                               city, "CITY_AGG", mix_a, mix_b, label, "city", theme_a, theme_b))
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
            theme_a, theme_b = theme_shares(a), theme_shares(b)
            i += 1
            items.append(make_item(f"task3_{city.lower().replace(' ', '')}_u{i:04d}",
                                   city, str(uid), mix_a, mix_b,
                                   decide_label(theme_a, theme_b), "user", theme_a, theme_b))
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