import re
import csv
import argparse

TRAIL_CANDIDATES = ["trail_id", "trail", "traj_id", "trajectory_id"]
USER_CANDIDATES = ["user_id", "user", "uid"]
CAT_CANDIDATES = ["venue_category", "category", "poi_category", "categories",
                  "venue_category_name", "category_name"]
TEXT_CANDIDATES = ["inputs", "input", "text", "prompt", "context"]

# matches "... which is a Pub, ..." / "... which is an Italian Restaurant, ..."
CAT_RE = re.compile(r"which is (?:a|an)\s+(.+?)\s*,", re.IGNORECASE)


def pick(colnames, candidates, override=None):
    if override:
        return override
    lower = {c.lower(): c for c in colnames}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def is_listish(v):
    return isinstance(v, (list, tuple))


def rows_from_record(rec, t, u, c, text_col):
    """Yield (trail_id, user_id, venue_category) tuples from one dataset row."""
    tid, uid = rec.get(t), rec.get(u)
    if c:  # explicit category column
        cat = rec.get(c)
        if is_listish(cat):
            for one in cat:
                if one:
                    yield tid, uid, one
        elif cat:
            yield tid, uid, cat
    elif text_col:  # parse categories out of the text field
        for cat in CAT_RE.findall(str(rec.get(text_col, ""))):
            yield tid, uid, cat.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--trail-col", default=None)
    ap.add_argument("--user-col", default=None)
    ap.add_argument("--cat-col", default=None)
    ap.add_argument("--text-col", default=None)
    args = ap.parse_args()

    from datasets import load_dataset, concatenate_datasets
    ds_id = args.dataset or f"CRUISEResearchGroup/Massive-STEPS-{args.city}"
    print(f"Loading {ds_id} ...")
    ds = load_dataset(ds_id)
    parts = [ds[s] for s in ds.keys()]
    data = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    cols = data.column_names
    print(f"Splits merged: {list(ds.keys())}  |  rows: {len(data)}  |  columns: {cols}")

    t = pick(cols, TRAIL_CANDIDATES, args.trail_col)
    u = pick(cols, USER_CANDIDATES, args.user_col)
    c = pick(cols, CAT_CANDIDATES, args.cat_col)
    text_col = pick(cols, TEXT_CANDIDATES, args.text_col)

    if args.inspect:
        print("\nFirst row:")
        for k, v in data[0].items():
            prev = v if not is_listish(v) else f"[list len={len(v)}] {list(v)[:5]}"
            print(f"  {k}: {str(prev)[:160]}")
        print(f"\nAuto-detected -> trail: {t}  user: {u}  category: {c}  text: {text_col}")
        if not c and text_col:
            sample = list(rows_from_record(data[0], t, u, c, text_col))
            print(f"Category parsing from '{text_col}' -> sample: {sample[:5]}")
        return

    if not (t and u and (c or text_col)):
        raise SystemExit(f"Could not resolve columns (trail={t}, user={u}, cat={c}, text={text_col}). "
                         f"Run --inspect and pass --trail-col/--user-col/--cat-col or --text-col.")

    out = args.out or f"data/{args.city.lower().replace(' ', '')}_checkins.csv"
    n = 0
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["trail_id", "user_id", "venue_category"])
        for rec in data:
            for tid, uid, cat in rows_from_record(rec, t, u, c, text_col):
                w.writerow([tid, uid, cat]); n += 1
    print(f"Wrote {n} check-in rows to {out}")
    print("Next: python src/user_overlap.py --checkins " + out)


if __name__ == "__main__":
    main()