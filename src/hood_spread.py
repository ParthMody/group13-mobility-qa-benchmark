import re
import argparse
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset

from src.hood_labels import PAIR, period_of, containers_for, norm

# identical to src/build_items.py -- keep in sync
THEME = {
    "home":      ["home", "residential", "apartment", "condo", "housing",
                  "neighborhood", "neighbourhood", "dorm", "house"],
    "work":      ["office", "coworking", "corporate", "cubicle", "conference room",
                  "workplace"],
    "transit":   ["train", "metro", "subway", "bus", "station", "platform", "tram",
                  "light rail", "airport", "terminal", "gate", "ferry", "boat",
                  "road", "bridge", "transport", "taxi", "parking", "highway"],
    "food":      ["cafe", "café", "coffee", "restaurant", "bar", "pub", "food",
                  "bakery", "diner", "deli", "pizza", "noodle", "ramen", "soba",
                  "sushi", "bistro", "brewery", "winery", "tea room", "dessert",
                  "eatery", "steakhouse", "grill", "snack"],
    "leisure":   ["cinema", "theater", "theatre", "gallery", "museum", "gym",
                  "fitness", "nightclub", "club", "stadium", "arena",
                  "entertainment", "arcade", "bowling", "spa", "concert"],
    "outdoors":  ["park", "beach", "waterfront", "trail", "garden", "mountain",
                  "outdoors", "lake", "river", "pier", "monument", "landmark",
                  "harbor", "harbour", "plaza", "scenic"],
    "education": ["university", "college", "school", "campus", "library",
                  "classroom", "lab", "student"],
    "civic":     ["church", "temple", "mosque", "synagogue", "shrine", "hospital",
                  "medical", "clinic", "government", "court", "police", "embassy",
                  "community", "civic", "county", "city"],
    "shopping":  ["mall", "shop", "store", "market", "boutique", "supermarket",
                  "grocery", "retail", "outlet", "pharmacy", "bookstore",
                  "department"],
}

NAVY, INDIGO, MINT, ROSE = "#0F172A", "#6366F1", "#14B8A6", "#E11D48"


def theme_of(cat):
    c = str(cat).lower()
    for t, kws in THEME.items():
        if any(k in c for k in kws):
            return t
    return "other"


def shares(counter):
    tot = sum(counter.values()) or 1
    return {k: v / tot for k, v in counter.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="New York")
    ap.add_argument("--min-n", type=int, default=60,
                    help="minimum check-ins per period for a locality to count")
    ap.add_argument("--keep-containers", action="store_true")
    ap.add_argument("--out", default="fig_hood_spread.pdf")
    args = ap.parse_args()

    ds = load_dataset(f"CRUISEResearchGroup/Massive-STEPS-{args.city.replace(' ', '-')}")
    rows = [r for s in ds for r in ds[s]]

    # locality -> period -> Counter(theme)
    hood = defaultdict(lambda: defaultdict(Counter))
    city = defaultdict(Counter)
    n_checkins = 0
    for r in rows:
        p = period_of(r.get("trail_id"))
        if p is None:
            continue
        for cat, loc in PAIR.findall(str(r.get("inputs", ""))):
            t = theme_of(cat)
            hood[loc.strip()][p][t] += 1
            city[p][t] += 1
            n_checkins += 1

    containers = containers_for(args.city, args.keep_containers)

    # keep localities with enough data in both periods
    keep = {}
    for loc, per in hood.items():
        a, b = sum(per["2013"].values()), sum(per["2018"].values())
        if a >= args.min_n and b >= args.min_n:
            if norm(loc) in containers:
                continue
            keep[loc] = (shares(per["2013"]), shares(per["2018"]), a, b)

    city_a, city_b = shares(city["2013"]), shares(city["2018"])
    city_delta = {t: (city_b.get(t, 0) - city_a.get(t, 0)) * 100
                  for t in set(city_a) | set(city_b)}

    print(f"\n=== {args.city} ===")
    print(f"  check-ins parsed        : {n_checkins:,}")
    print(f"  localities              : {len(hood):,}")
    print(f"  kept (>= {args.min_n} per period) : {len(keep):,}"
          + (f"   [excluded containers: {', '.join(sorted(containers))}]" if containers else ""))
    if len(keep) < 3:
        print("\n  Too few units to say anything about spread. Stopping.")
        return

    # per-theme spread across neighbourhoods vs the city aggregate
    themes = [t for t, _ in sorted(city_delta.items(), key=lambda x: -abs(x[1]))
              if t != "other"][:6]

    print(f"\n  {'theme':<11} {'city':>7}   {'hood min':>9} {'hood max':>9} "
          f"{'range':>7}   {'opposite sign':>13}")
    print("  " + "-" * 66)
    spread = {}
    for t in themes:
        ds_ = [( (b.get(t, 0) - a.get(t, 0)) * 100 ) for a, b, _, _ in keep.values()]
        spread[t] = ds_
        cd = city_delta.get(t, 0)
        opp = sum(1 for d in ds_ if (d > 0) != (cd > 0))
        print(f"  {t:<11} {cd:>+7.1f}   {min(ds_):>+9.1f} {max(ds_):>+9.1f} "
              f"{max(ds_)-min(ds_):>7.1f}   {opp:>5}/{len(ds_):<7}")

    # headline stat on the city's own labelled theme
    top = themes[0]
    cd = city_delta[top]
    opp = sum(1 for d in spread[top] if (d > 0) != (cd > 0))
    print(f"\n  Headline: the city moves {cd:+.1f}pp on '{top}', but across "
          f"{len(keep)} neighbourhoods\n  that delta ranges "
          f"{min(spread[top]):+.1f} to {max(spread[top]):+.1f}pp, and {opp} move "
          f"in the opposite\n  direction to the city aggregate.")

    # ---- figure: strip plot, neighbourhood deltas vs city aggregate ----
    fig, ax = plt.subplots(figsize=(9, 0.62 * len(themes) + 1.6))
    for i, t in enumerate(themes):
        ds_ = spread[t]
        y = [i + (j % 5 - 2) * 0.045 for j in range(len(ds_))]   # jitter
        cd = city_delta.get(t, 0)
        cols = [ROSE if (d > 0) != (cd > 0) else INDIGO for d in ds_]
        ax.scatter(ds_, y, s=22, c=cols, alpha=.75, linewidths=0, zorder=3)
        ax.scatter([cd], [i], s=190, marker="|", c=NAVY, zorder=4, linewidths=2.2)

    ax.axvline(0, color=NAVY, lw=.8, zorder=1)
    ax.set_yticks(range(len(themes)), themes, fontsize=10)
    ax.set_xlabel("change in theme share, 2013 $\\rightarrow$ 2018 (pp)", fontsize=10)
    ax.set_title(f"{args.city}: neighbourhood-level change vs the city aggregate",
                 fontsize=11, color=NAVY, loc="left")
    ax.grid(axis="x", lw=.4, alpha=.35)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=INDIGO, label="neighbourhood (same direction as city)"),
        Line2D([], [], marker="o", ls="", color=ROSE, label="neighbourhood (opposite direction)"),
        Line2D([], [], marker="|", ls="", color=NAVY, markersize=12, label="city aggregate"),
    ], frameon=False, fontsize=8.5, loc="lower right")

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"\n  wrote {args.out}")

    print(f"\n  {'neighbourhood':<24} {'2013':>7} {'2018':>7}   "
          + " ".join(f"{t[:6]:>7}" for t in themes[:4]))
    print("  " + "-" * (24 + 18 + 8 * 4))
    for loc, (a, b, na, nb) in sorted(keep.items(), key=lambda x: -(x[1][2] + x[1][3]))[:12]:
        ds_ = [f"{(b.get(t,0)-a.get(t,0))*100:>+7.1f}" for t in themes[:4]]
        print(f"  {loc[:24]:<24} {na:>7,} {nb:>7,}   " + " ".join(ds_))


if __name__ == "__main__":
    main()