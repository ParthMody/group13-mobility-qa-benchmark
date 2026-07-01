import argparse
import csv
from collections import defaultdict


def period_of(trail_id):
    s = str(trail_id)
    if s.startswith("2013_"):
        return "2013"
    if s.startswith("2018_"):
        return "2018"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkins", required=True)
    args = ap.parse_args()

    counts = defaultdict(lambda: {"2013": 0, "2018": 0})
    with open(args.checkins, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = period_of(r.get("trail_id"))
            if p:
                counts[r["user_id"]][p] += 1

    total = len(counts)
    both1 = sum(1 for c in counts.values() if c["2013"] and c["2018"])
    both5 = sum(1 for c in counts.values() if c["2013"] >= 5 and c["2018"] >= 5)
    both10 = sum(1 for c in counts.values() if c["2013"] >= 10 and c["2018"] >= 10)
    only13 = sum(1 for c in counts.values() if c["2013"] and not c["2018"])
    only18 = sum(1 for c in counts.values() if c["2018"] and not c["2013"])

    print("\n=== 2013 vs 2018 user overlap ===")
    print(f"  total users seen        : {total}")
    print(f"  2013 only               : {only13}")
    print(f"  2018 only               : {only18}")
    print(f"  in BOTH (>=1 each)       : {both1}")
    print(f"  in BOTH (>=5 each)       : {both5}   <- build_items.py --unit user uses this")
    print(f"  in BOTH (>=10 each)      : {both10}")

    print("\n=== recommendation ===")
    if both5 >= 40:
        print(f"  {both5} usable per-user pairs -> PER-USER ML is viable.")
        print("  -> build_items.py --unit user, and run the ML track.")
    elif both5 >= 15:
        print(f"  {both5} per-user pairs is thin. Options:")
        print("   - per-user ML as a weak/secondary result, OR")
        print("   - pull 2-3 more cities and pool per-user pairs, OR")
        print("   - make the zero-shot LLM result the headline.")
    else:
        print(f"  Only {both5} per-user pairs -> PER-USER ML NOT viable.")
        print("  -> Use per-city items across several cities for coverage, and let the")
        print("     ZERO-SHOT LLM track be the headline result. (This is a fine outcome —")
        print("     scarce labeled data is exactly the case zero-shot is meant for.)")


if __name__ == "__main__":
    main()