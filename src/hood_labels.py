import re
import json
import argparse
import unicodedata
from collections import Counter, defaultdict

from datasets import load_dataset

from src.build_items import theme_of, decide_label, LABEL_SET

PAIR = re.compile(r"which is (?:a|an)\s+(.+?),\s*at\s+([^,]+?),\s*[A-Z]{2}\b")

# A locality is not a sub-unit of the city if it bears the city's own name, or if
# it is a well-known administrative container of the neighbourhoods beneath it.
# The first case is a rule; the second is an explicit, stated list.
CONTAINER_EXTRA = {
    "new york": {"manhattan", "brooklyn", "queens", "the bronx", "bronx",
                 "staten island"},
}


def norm(s):
    """Lowercase and strip diacritics, so 'Sao Paulo' matches 'Sao Paulo'
    and 'Uskudar' matches 'Uskudar' regardless of accents."""
    d = unicodedata.normalize("NFKD", str(s).lower())
    return "".join(c for c in d if not unicodedata.combining(c)).strip()


def containers_for(city, keep=False):
    if keep:
        return set()
    c = norm(city)
    auto = {c, c + " city", c.replace(" city", "")}
    return {norm(x) for x in auto | CONTAINER_EXTRA.get(c, set())}


def period_of(trail_id):
    s = str(trail_id)
    return "2013" if s.startswith("2013_") else ("2018" if s.startswith("2018_") else None)


def shares(counter):
    tot = sum(counter.values()) or 1
    return {k: v / tot for k, v in counter.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="New York")
    ap.add_argument("--min-n", type=int, default=60)
    ap.add_argument("--keep-containers", action="store_true")
    ap.add_argument("--out", default="data/task3_neighbourhood_analysis.json")
    args = ap.parse_args()

    ds = load_dataset(f"CRUISEResearchGroup/Massive-STEPS-{args.city.replace(' ', '-')}")
    rows = [r for s in ds for r in ds[s]]

    hood = defaultdict(lambda: defaultdict(Counter))   # locality -> period -> theme counts
    city = defaultdict(Counter)                        # period -> theme counts
    for r in rows:
        p = period_of(r.get("trail_id"))
        if p is None:
            continue
        for cat, loc in PAIR.findall(str(r.get("inputs", ""))):
            t = theme_of(cat)
            hood[loc.strip()][p][t] += 1
            city[p][t] += 1

    containers = containers_for(args.city, args.keep_containers)

    # the city's own label, by the same rule Task 3 uses
    city_label = decide_label(shares(city["2013"]), shares(city["2018"]))

    units = []
    for loc, per in sorted(hood.items()):
        a, b = sum(per["2013"].values()), sum(per["2018"].values())
        if a < args.min_n or b < args.min_n or norm(loc) in containers:
            continue
        sa, sb = shares(per["2013"]), shares(per["2018"])
        label = decide_label(sa, sb)
        deltas = {t: round((sb.get(t, 0) - sa.get(t, 0)) * 100, 1)
                  for t in set(sa) | set(sb)}
        units.append({
            "neighbourhood": loc, "n_2013": a, "n_2018": b,
            "label": label,
            "theme_deltas": dict(sorted(deltas.items(), key=lambda x: -abs(x[1]))),
        })

    if len(units) < 3:
        print(f"Only {len(units)} units at --min-n {args.min_n}. Nothing to report.")
        return

    dist = Counter(u["label"] for u in units)
    agree = sum(1 for u in units if u["label"] == city_label)

    print(f"\n=== {args.city}: Task 3's rule at neighbourhood level ===")
    print(f"  city label (whole-city aggregate) : {city_label}")
    dropped = sorted(l for l in hood if norm(l) in containers)
    print(f"  neighbourhoods (>= {args.min_n} per period) : {len(units)}")
    if dropped:
        print(f"  containers excluded : {', '.join(dropped)}")
    print(f"\n  label distribution across neighbourhoods:")
    for lab, n in dist.most_common():
        bar = "#" * n
        print(f"    {lab:<24} {n:>3}  {bar}")
    print(f"\n  distinct labels : {len(dist)} of {len(LABEL_SET)} available")
    print(f"  agree with city : {agree}/{len(units)} ({agree/len(units):.0%})")

    print(f"\n  {'neighbourhood':<24} {'2013':>7} {'2018':>7}  {'label':<24} same?")
    print("  " + "-" * 72)
    for u in sorted(units, key=lambda x: -(x["n_2013"] + x["n_2018"])):
        same = "yes" if u["label"] == city_label else "NO"
        print(f"  {u['neighbourhood'][:24]:<24} {u['n_2013']:>7,} {u['n_2018']:>7,}"
              f"  {u['label']:<24} {same}")

    # concentration: how much of the city is a handful of neighbourhoods?
    tot = sum(u["n_2013"] + u["n_2018"] for u in units)
    top2 = sorted(units, key=lambda x: -(x["n_2013"] + x["n_2018"]))[:2]
    share2 = sum(u["n_2013"] + u["n_2018"] for u in top2) / tot
    print(f"\n  top 2 neighbourhoods ({', '.join(u['neighbourhood'] for u in top2)}) "
          f"hold {share2:.0%}\n  of the check-ins across these {len(units)} units.")

    payload = {
        "city": args.city, "city_label": city_label, "min_n": args.min_n,
        "n_units": len(units), "label_distribution": dict(dist),
        "agree_with_city": agree, "units": units,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()