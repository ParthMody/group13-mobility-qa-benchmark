import re
import argparse
from collections import Counter, defaultdict

from datasets import load_dataset

CATEGORY_RE = re.compile(r"which is (?:a|an)\s+(.+?)\s*,", re.IGNORECASE)
LOCALITY_RE = re.compile(r"\bat\s+([^,]{2,60}?),\s*[A-Z]{2}\b")


def load_city(city):
    slug = city.replace(" ", "-")
    ds = load_dataset(f"CRUISEResearchGroup/Massive-STEPS-{slug}")
    # concatenate whatever splits exist
    rows = []
    for split in ds:
        rows.extend(ds[split])
    return rows


def period_of(trail_id):
    s = str(trail_id)
    if s.startswith("2013_"):
        return "2013"
    if s.startswith("2018_"):
        return "2018"
    return None


def inspect(rows, n=5):
    print(f"\n--- raw records ({n} of {len(rows):,}) ---\n")
    for r in rows[:n]:
        text = str(r.get("inputs", ""))
        print("trail_id:", r.get("trail_id"))
        print("inputs  :", text[:400].replace("\n", " "))
        cat = CATEGORY_RE.search(text)
        loc = LOCALITY_RE.search(text)
        print("  -> category:", cat.group(1).strip() if cat else "NO MATCH")
        print("  -> locality:", loc.group(1).strip() if loc else "NO MATCH")
        print()
    # how often do the patterns hit across the whole city?
    cat_hits = sum(1 for r in rows if CATEGORY_RE.search(str(r.get("inputs", ""))))
    loc_hits = sum(1 for r in rows if LOCALITY_RE.search(str(r.get("inputs", ""))))
    print(f"category regex matches: {cat_hits:,}/{len(rows):,} "
          f"({cat_hits/max(len(rows),1):.0%})")
    print(f"locality regex matches: {loc_hits:,}/{len(rows):,} "
          f"({loc_hits/max(len(rows),1):.0%})")
    print("\nIf locality is far below ~80%, the regex needs adjusting before "
          "the counts below mean anything.")


def probe(rows, city, thresholds=(10, 20, 30, 50, 100)):
    # locality -> {period: count}
    units = defaultdict(Counter)
    unparsed = 0
    for r in rows:
        p = period_of(r.get("trail_id"))
        if p is None:
            continue
        text = str(r.get("inputs", ""))
        m = LOCALITY_RE.search(text)
        if not m:
            unparsed += 1
            continue
        units[m.group(1).strip()][p] += 1

    per = Counter(period_of(r.get("trail_id")) for r in rows)
    print(f"\n=== {city} ===")
    print(f"  records          : {len(rows):,}  (2013={per.get('2013',0):,}, "
          f"2018={per.get('2018',0):,})")
    print(f"  localities found : {len(units):,}   (unparsed rows: {unparsed:,})")

    both = [l for l, c in units.items() if c["2013"] > 0 and c["2018"] > 0]
    print(f"  with BOTH periods (any count): {len(both):,}")

    print(f"\n  {'min check-ins per period':<26} {'units surviving':>15}")
    print("  " + "-" * 42)
    for t in thresholds:
        n = sum(1 for l, c in units.items() if c["2013"] >= t and c["2018"] >= t)
        print(f"  {'>= ' + str(t):<26} {n:>15}")

    # show the biggest surviving units at a moderate threshold
    strong = sorted(
        ((l, c["2013"], c["2018"]) for l, c in units.items()
         if c["2013"] >= 30 and c["2018"] >= 30),
        key=lambda x: -(x[1] + x[2]))
    if strong:
        print(f"\n  top surviving units at >=30 (showing up to 10):")
        for l, a, b in strong[:10]:
            print(f"    {l[:38]:<38} 2013={a:>6,}  2018={b:>6,}")
    else:
        print("\n  no units survive at >=30.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--inspect", action="store_true",
                    help="print raw records and regex hit-rates, no counting")
    args = ap.parse_args()

    rows = load_city(args.city)
    if args.inspect:
        inspect(rows)
    else:
        probe(rows, args.city)


if __name__ == "__main__":
    main()