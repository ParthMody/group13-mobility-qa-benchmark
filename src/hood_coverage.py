import re
import json
import argparse
from collections import Counter, defaultdict

from datasets import load_dataset

from src.hood_labels import PAIR, period_of, containers_for, norm

CITIES = ["New York", "Kuwait City", "Istanbul", "Sydney", "Melbourne",
          "Moscow", "Jakarta", "Petaling Jaya", "Sao Paulo"]

MIN_UNITS = 3   # below this, a "distribution" is not a distribution


def analyse(city, min_n):
    ds = load_dataset(f"CRUISEResearchGroup/Massive-STEPS-{city.replace(' ', '-')}")
    rows = [r for s in ds for r in ds[s]]

    hood = defaultdict(Counter)          # locality -> period -> n check-ins
    for r in rows:
        p = period_of(r.get("trail_id"))
        if p is None:
            continue
        for _cat, loc in PAIR.findall(str(r.get("inputs", ""))):
            hood[loc.strip()][p] += 1

    containers = containers_for(city)
    total = len(hood)
    subs = {l: c for l, c in hood.items() if norm(l) not in containers}
    both = sum(1 for c in subs.values() if c["2013"] > 0 and c["2018"] > 0)
    clear = sorted(l for l, c in subs.items()
                   if c["2013"] >= min_n and c["2018"] >= min_n)

    if total <= 1:
        status = "no sub-units"
    elif len(clear) < MIN_UNITS:
        status = "sparse coverage"
    else:
        status = "analysed"

    return {"city": city, "localities": total, "sub_units": len(subs),
            "both_periods": both, "clearing": len(clear), "status": status}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-n", type=int, default=200)
    ap.add_argument("--out", default="data/hood_coverage.json")
    args = ap.parse_args()

    import pathlib
    data = pathlib.Path("data")

    # pull label distributions from any analyses already on disk
    dists = {}
    for p in list(data.glob("hood_*.json")) + [data / "task3_neighbourhood_analysis.json"]:
        if p.name == "hood_coverage.json" or not p.exists():
            continue
        try:
            an = json.load(open(p, encoding="utf-8"))
            if "label_distribution" in an:
                dists[an["city"]] = an
        except Exception:
            continue

    rows = []
    for c in CITIES:
        try:
            r = analyse(c, args.min_n)
        except Exception as e:
            print(f"  {c}: skipped ({type(e).__name__})")
            continue
        an = dists.get(c)
        if an and r["status"] == "analysed":
            r["n_labels"] = len(an["label_distribution"])
            r["agree"] = f'{an["agree_with_city"]}/{an["n_units"]}'
            r["city_label"] = an["city_label"]
        rows.append(r)
        print(f"  {c:<15} localities={r['localities']:<4} "
              f"clearing>={args.min_n}: {r['clearing']:<3} -> {r['status']}")

    print(f"\n  Locality-field coverage, threshold >= {args.min_n} check-ins per period\n")
    print(f"  {'City':<15}{'localities':>11}{'both periods':>14}{'clearing':>10}"
          f"{'labels':>8}{'agree':>8}   status")
    print("  " + "-" * 78)
    for r in rows:
        lab = str(r.get("n_labels", "--"))
        agr = r.get("agree", "--")
        print(f"  {r['city']:<15}{r['localities']:>11}{r['both_periods']:>14}"
              f"{r['clearing']:>10}{lab:>8}{agr:>8}   {r['status']}")

    n_an = sum(1 for r in rows if r["status"] == "analysed")
    n_sp = sum(1 for r in rows if r["status"] == "sparse coverage")
    n_no = sum(1 for r in rows if r["status"] == "no sub-units")
    print(f"\n  analysed: {n_an}   sparse coverage: {n_sp}   no sub-units: {n_no}")

    json.dump({"min_n": args.min_n, "cities": rows},
              open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()