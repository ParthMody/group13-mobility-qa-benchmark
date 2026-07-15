import json
import random
import argparse
from collections import Counter

MARGIN = 0.005    # 0.5pp -- below this an "extremum" question has no clear answer
MOVE = 0.01       # 1pp   -- a category must move at least this much to be asked about
GAP = 0.02        # 2pp   -- minimum gap between two deltas for a comparison


def render(mix_a, mix_b):
    def fmt(m):
        return ", ".join(f"{k} {v*100:.1f}%" for k, v in
                         sorted(m.items(), key=lambda x: -x[1]))
    return (f"Share of check-ins by POI category (the most visited categories in "
            f"each period).\n2013: {fmt(mix_a)}\n2018: {fmt(mix_b)}")


def item(qid, city, ctx, question, choices, answer, rationale, probe, tier):
    return {
        "question_id": qid, "task": "task3_change_probe", "city": city,
        "user_id": "CITY_AGG", "context_sequence": [], "context_text": ctx,
        "target_time": "2013_vs_2018", "question": question,
        "choices": choices, "answer": answer, "rationale": rationale,
        "source_dataset": "massive_steps",
        "metadata": {"answer_type": "closed", "eval_mode": "classification",
                     "probe_type": probe, "tier": tier,
                     "difficulty": {"lookup": "easy", "change": "easy",
                                    "relational": "medium"}[tier],
                     "unit": "city", "derived_from": "displayed_category_mix"},
    }


def build_city(it, rng, idx):
    city = it["city"]
    mix_a = it["context_sequence"][0]["category_mix"]
    mix_b = it["context_sequence"][1]["category_mix"]
    ctx = render(mix_a, mix_b)
    # Delta questions are restricted to categories listed in BOTH years. A
    # category shown in one list and absent from the other has an unknown share
    # in that other period -- it may simply have been pushed out of the top-8 --
    # so its delta cannot be computed from the display, only bounded. Including
    # those would produce gold answers the model cannot derive.
    cats = sorted(set(mix_a) & set(mix_b))
    delta = {c: mix_b[c] - mix_a[c] for c in cats}
    ranked = sorted(delta.items(), key=lambda x: -x[1])
    slug = city.lower().replace(" ", "")
    flip = (idx % 2 == 0)          # alternate binary answers across cities
    out = []

    def opts(correct, pool=None):
        pool = [c for c in (pool or cats) if c != correct]
        return sorted([correct] + rng.sample(pool, min(3, len(pool))))

    def add(suffix, q, choices, ans, why, probe, tier):
        out.append(item(f"task3probe_{slug}_{suffix}", city, ctx, q,
                        choices, ans, why, probe, tier))

    # ---------- tier: lookup ----------
    top_a = max(mix_a, key=mix_a.get)
    if len(mix_a) > 1:
        second = sorted(mix_a.values(), reverse=True)[1]
        if mix_a[top_a] - second >= MARGIN:
            add("top2013",
                "Of the categories listed for 2013, which had the largest share of check-ins?",
                opts(top_a, list(mix_a)), top_a,
                f"{top_a} is the largest 2013 share at {mix_a[top_a]*100:.1f}%.",
                "top_2013", "lookup")

    top_b = max(mix_b, key=mix_b.get)
    if len(mix_b) > 1:
        second = sorted(mix_b.values(), reverse=True)[1]
        if mix_b[top_b] - second >= MARGIN:
            add("top2018",
                "Of the categories listed for 2018, which had the largest share of check-ins?",
                opts(top_b, list(mix_b)), top_b,
                f"{top_b} is the largest 2018 share at {mix_b[top_b]*100:.1f}%.",
                "top_2018", "lookup")

    only_b = [c for c in mix_b if c not in mix_a]
    if only_b:
        c = rng.choice(only_b)
        pool = [x for x in cats if x in mix_a]     # distractors that ARE in 2013
        if len(pool) >= 3:
            add("new2018",
                "Which of these categories is listed for 2018 but not for 2013?",
                opts(c, pool + [c]), c,
                f"{c} appears in the 2018 list ({mix_b[c]*100:.1f}%) and not in the 2013 list.",
                "listed_only_2018", "lookup")

    # ---------- tier: change ----------
    ups = [c for c in cats if delta[c] >= MOVE]
    downs = [c for c in cats if delta[c] <= -MOVE]
    pool = (ups if flip else downs) or (downs if flip else ups)
    if pool:
        c = rng.choice(pool)
        add("direction",
            f"Comparing the two lists, did the share of check-ins at {c} rise or fall?",
            ["rose", "fell"], "rose" if delta[c] > 0 else "fell",
            f"{c}: {mix_a.get(c,0)*100:.1f}% -> {mix_b.get(c,0)*100:.1f}% ({delta[c]*100:+.1f}pp).",
            "direction", "change")

    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] >= MARGIN and ranked[0][1] > 0:
        c, d = ranked[0]
        add("riser",
            "Of the categories listed in both years, which gained the most share?",
            opts(c), c,
            f"{c} gains {d*100:+.1f}pp, ahead of {ranked[1][0]} at {ranked[1][1]*100:+.1f}pp.",
            "biggest_riser", "change")

    if len(ranked) > 1 and ranked[-2][1] - ranked[-1][1] >= MARGIN and ranked[-1][1] < 0:
        c, d = ranked[-1]
        add("faller",
            "Of the categories listed in both years, which lost the most share?",
            opts(c), c,
            f"{c} loses {d*100:+.1f}pp, more than {ranked[-2][0]} at {ranked[-2][1]*100:+.1f}pp.",
            "biggest_faller", "change")

    # ---------- tier: relational ----------
    pairs = [(x, y) for i, x in enumerate(cats) for y in cats[i+1:]
             if abs(delta[x] - delta[y]) >= GAP]
    if pairs:
        x, y = rng.choice(pairs)
        if (delta[x] > delta[y]) != flip:
            x, y = y, x
        add("compare",
            f"Did {x} gain more share than {y} between 2013 and 2018? "
            f"(A loss counts as gaining less.)",
            ["yes", "no"], "yes" if delta[x] > delta[y] else "no",
            f"{x} moves {delta[x]*100:+.1f}pp; {y} moves {delta[y]*100:+.1f}pp.",
            "comparison", "relational")

    # both_rose -- pick a pair that yields the intended answer
    want_yes = not flip
    cands = ([(x, y) for x in ups for y in ups if x < y] if want_yes else
             [(x, y) for x in ups for y in downs])
    if cands:
        x, y = rng.choice(cands)
        ans = "yes" if (delta[x] > 0 and delta[y] > 0) else "no"
        add("bothrose",
            f"Did the share of check-ins rise for both {x} and {y}?",
            ["yes", "no"], ans,
            f"{x} {delta[x]*100:+.1f}pp; {y} {delta[y]*100:+.1f}pp.",
            "both_rose", "relational")

    stable = sorted(cats, key=lambda c: abs(delta[c]))
    if len(stable) > 1 and abs(delta[stable[1]]) - abs(delta[stable[0]]) >= MARGIN:
        c = stable[0]
        add("stable",
            "Of the categories listed in both years, which share changed the least?",
            opts(c), c,
            f"{c} moves only {delta[c]*100:+.1f}pp; next closest is {stable[1]} "
            f"at {delta[stable[1]]*100:+.1f}pp.",
            "most_stable", "relational")

    n_up = sum(1 for c in cats if delta[c] > 0)
    distract = sorted({max(0, n_up - 2), max(0, n_up - 1), n_up + 1, n_up + 2,
                       len(cats)} - {n_up})
    if len(distract) >= 3:
        add("countrisers",
            "Of the categories listed in both years, how many gained share?",
            sorted([str(n_up)] + [str(d) for d in rng.sample(distract, 3)], key=int),
            str(n_up),
            f"{n_up} of {len(cats)} listed categories have a higher 2018 share.",
            "count_risers", "relational")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/task3_items.jsonl")
    ap.add_argument("--out", default="data/task3probe_items.jsonl")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    src = [json.loads(l) for l in open(args.items, encoding="utf-8") if l.strip()]

    all_items = []
    for idx, it in enumerate(src):
        got = build_city(it, rng, idx)
        all_items.extend(got)
        print(f"  {it['city']:<15} {len(got):>2} probes")

    with open(args.out, "w", encoding="utf-8") as f:
        for i in all_items:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")

    print(f"\n  {len(all_items)} probes from {len(src)} cities -> {args.out}")
    print("  by tier :", dict(Counter(i["metadata"]["tier"] for i in all_items)))
    print("  by type :", dict(Counter(i["metadata"]["probe_type"] for i in all_items)))
    binaries = Counter(i["answer"] for i in all_items
                       if i["answer"] in ("rose", "fell", "yes", "no"))
    print("  binary answer key:", dict(binaries))
    print(f"\n  Report as {len(src)} cities x ~{round(len(all_items)/len(src))} probes "
          "(clustered), not as independent items.")


if __name__ == "__main__":
    main()