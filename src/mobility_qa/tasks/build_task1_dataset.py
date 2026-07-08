"""Build dataset-backed Task 1 QA records from Massive-STEPS check-ins."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import ssl
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import certifi

from mobility_qa.io import write_csv_records, write_jsonl
from mobility_qa.schema import validate_qa_records


TASK_NAME = "task1_next_poi_category"
SOURCE_DATASET = "massive_steps"
VERIFICATION_STATUS = "verified_massive_steps_taxonomy_mapping"

TAXONOMY_URL = (
    "https://gist.githubusercontent.com/missinglink/"
    "a0344050d3e2b52256d7/raw/bd2bae3b17f85de544caa53daaff36a4eba8ad2b/"
    "foursquare-taxonomy.json"
)
HF_DATASET_PREFIX = "https://huggingface.co/datasets/CRUISEResearchGroup"

CITY_SPECS = {
    "bandung": ("Bandung", "Massive-STEPS-Bandung", "bandung"),
    "beijing": ("Beijing", "Massive-STEPS-Beijing", "beijing"),
    "istanbul": ("Istanbul", "Massive-STEPS-Istanbul", "istanbul"),
    "jakarta": ("Jakarta", "Massive-STEPS-Jakarta", "jakarta"),
    "kuwait_city": (
        "Kuwait City",
        "Massive-STEPS-Kuwait-City",
        "kuwait_city",
    ),
    "melbourne": ("Melbourne", "Massive-STEPS-Melbourne", "melbourne"),
    "moscow": ("Moscow", "Massive-STEPS-Moscow", "moscow"),
    "new_york": ("New York", "Massive-STEPS-New-York", "new_york"),
    "palembang": ("Palembang", "Massive-STEPS-Palembang", "palembang"),
    "petaling_jaya": (
        "Petaling Jaya",
        "Massive-STEPS-Petaling-Jaya",
        "petaling_jaya",
    ),
    "sao_paulo": ("São Paulo", "Massive-STEPS-Sao-Paulo", "sao_paulo"),
    "shanghai": ("Shanghai", "Massive-STEPS-Shanghai", "shanghai"),
    "sydney": ("Sydney", "Massive-STEPS-Sydney", "sydney"),
    "tangerang": ("Tangerang", "Massive-STEPS-Tangerang", "tangerang"),
    "tokyo": ("Tokyo", "Massive-STEPS-Tokyo", "tokyo"),
}

BROAD_CATEGORIES = (
    "Arts & Entertainment",
    "College & University",
    "Event",
    "Food",
    "Nightlife Spot",
    "Outdoors & Recreation",
    "Professional & Other Places",
    "Residence",
    "Shop & Service",
    "Travel & Transport",
)

REQUIRED_CHECKIN_FIELDS = {
    "trail_id",
    "venue_category",
    "venue_category_id",
    "timestamp",
}


def _download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "mobility-qa/1.0"})
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(
        request, context=ssl_context
    ) as response, temporary_path.open("wb") as file:
        shutil.copyfileobj(response, file)
    temporary_path.replace(output_path)


def massive_steps_url(city_key: str, split: str) -> str:
    """Return the official Hugging Face CSV URL for one city split."""
    _, dataset_name, file_prefix = CITY_SPECS[city_key]
    filename = f"{file_prefix}_checkins_{split}.csv"
    return f"{HF_DATASET_PREFIX}/{dataset_name}/resolve/main/{filename}?download=true"


def download_sources(
    raw_dir: str | Path,
    city_keys: Iterable[str],
    split: str,
    *,
    force: bool = False,
) -> tuple[Path, dict[str, Path]]:
    """Download the taxonomy and official Massive-STEPS city CSV files."""
    raw_dir = Path(raw_dir)
    taxonomy_path = raw_dir / "foursquare-taxonomy.json"
    if force or not taxonomy_path.exists():
        _download_file(TAXONOMY_URL, taxonomy_path)

    checkin_paths = {}
    for city_key in city_keys:
        _, _, file_prefix = CITY_SPECS[city_key]
        output_path = raw_dir / f"{file_prefix}_checkins_{split}.csv"
        if force or not output_path.exists():
            _download_file(massive_steps_url(city_key, split), output_path)
        checkin_paths[city_key] = output_path
    return taxonomy_path, checkin_paths


def flatten_foursquare_taxonomy(
    taxonomy: list[Mapping[str, Any]],
) -> dict[str, str]:
    """Map every legacy Foursquare category ID to its top-level category."""
    mapping: dict[str, str] = {}

    def visit(nodes: Iterable[Mapping[str, Any]], broad_category: str) -> None:
        for node in nodes:
            category_id = str(node.get("id", "")).strip()
            if category_id:
                mapping.setdefault(category_id, broad_category)
            children = node.get("categories", [])
            if isinstance(children, list):
                visit(children, broad_category)

    for root in taxonomy:
        broad_category = str(root.get("name", "")).strip()
        if broad_category not in BROAD_CATEGORIES:
            continue
        visit([root], broad_category)
    return mapping


def load_taxonomy_mapping(path: str | Path) -> dict[str, str]:
    """Load and flatten a historical Foursquare taxonomy JSON file."""
    with Path(path).open("r", encoding="utf-8") as file:
        taxonomy = json.load(file)
    if not isinstance(taxonomy, list):
        raise ValueError("Foursquare taxonomy must be a JSON list.")
    mapping = flatten_foursquare_taxonomy(taxonomy)
    if not mapping:
        raise ValueError("Foursquare taxonomy did not contain known broad categories.")
    return mapping


def _stable_number(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _difficulty(history: list[str], target: str) -> str:
    if history[-1] == target:
        return "easy"
    if target in history:
        return "medium"
    return "hard"


def _build_options(
    question_id: str,
    target: str,
    previous_category: str,
    transition_counts: Mapping[str, Counter[str]],
    global_counts: Counter[str],
    seed: int,
) -> list[str]:
    distractors: list[str] = []
    ranked = [
        category
        for category, _ in transition_counts.get(previous_category, Counter()).most_common()
    ]
    ranked.extend(category for category, _ in global_counts.most_common())
    ranked.extend(BROAD_CATEGORIES)

    for category in ranked:
        if category != target and category not in distractors:
            distractors.append(category)
        if len(distractors) == 3:
            break

    options = [target, *distractors]
    random.Random(_stable_number(question_id, seed)).shuffle(options)
    return options


def _format_context(city: str, history: pd.DataFrame, target_time: str) -> str:
    checkins = "; ".join(
        f"{row.timestamp}: {row.venue_category}"
        for row in history[["timestamp", "venue_category"]].itertuples(index=False)
    )
    return (
        f"The user's recent check-ins in {city} are {checkins}. "
        f"The target time is {target_time}. Which broad POI category is most likely next?"
    )


def _balanced_sample(
    records: list[dict[str, Any]], limit: int, seed: int
) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[record["correct_answer"]].append(record)
    for bucket in buckets.values():
        bucket.sort(key=lambda record: _stable_number(record["question_id"], seed))

    selected: list[dict[str, Any]] = []
    categories = sorted(buckets)
    while len(selected) < limit:
        added = False
        for category in categories:
            if buckets[category]:
                selected.append(buckets[category].pop())
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
    return sorted(selected, key=lambda record: record["question_id"])


def build_task1_records(
    checkins: pd.DataFrame,
    city: str,
    city_key: str,
    split: str,
    taxonomy_mapping: Mapping[str, str],
    *,
    limit: int = 200,
    min_history: int = 2,
    max_history: int = 6,
    seed: int = 5925,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build deterministic category-level QA records for one city."""
    missing = REQUIRED_CHECKIN_FIELDS.difference(checkins.columns)
    if missing:
        raise ValueError(f"Missing Massive-STEPS columns: {', '.join(sorted(missing))}")
    if min_history < 1 or max_history < min_history:
        raise ValueError("History limits must satisfy 1 <= min_history <= max_history.")

    frame = checkins.copy()
    frame["venue_category_id"] = frame["venue_category_id"].astype(str)
    frame["timestamp"] = frame["timestamp"].astype(str)
    frame = frame.sort_values(["trail_id", "timestamp"], kind="stable")

    trajectories: list[tuple[str, pd.DataFrame, list[str]]] = []
    transition_counts: dict[str, Counter[str]] = defaultdict(Counter)
    global_counts: Counter[str] = Counter()
    skipped_short = 0
    skipped_unmapped = 0

    for trail_id, group in frame.groupby("trail_id", sort=False):
        if len(group) < min_history + 1:
            skipped_short += 1
            continue
        broad = [taxonomy_mapping.get(category_id) for category_id in group["venue_category_id"]]
        if any(category is None for category in broad):
            skipped_unmapped += 1
            continue
        mapped = [str(category) for category in broad]
        trajectories.append((str(trail_id), group, mapped))
        global_counts.update(mapped)
        for previous, following in zip(mapped, mapped[1:]):
            transition_counts[previous][following] += 1

    candidates = []
    for trail_id, group, mapped in trajectories:
        target_row = group.iloc[-1]
        history = group.iloc[:-1].tail(max_history)
        history_broad = mapped[-(len(history) + 1) : -1]
        target_broad = mapped[-1]
        safe_trail_id = re.sub(r"[^A-Za-z0-9_-]+", "-", trail_id)
        question_id = f"task1_ms_{split}_{city_key}_{safe_trail_id}"
        options = _build_options(
            question_id,
            target_broad,
            history_broad[-1],
            transition_counts,
            global_counts,
            seed,
        )
        record = {
            "question_id": question_id,
            "task": TASK_NAME,
            "city": city,
            "context": _format_context(city, history, str(target_row["timestamp"])),
            "answer_type": "multiple choice",
            "options": options,
            "correct_answer": target_broad,
            "reasoning": (
                f"The held-out Massive-STEPS target category is "
                f"{target_row['venue_category']} ({target_row['venue_category_id']}), "
                f"which maps to the Foursquare top-level category {target_broad}."
            ),
            "difficulty": _difficulty(history_broad, target_broad),
            "source_dataset": SOURCE_DATASET,
            "verification_status": VERIFICATION_STATUS,
        }
        candidates.append(record)

    selected = _balanced_sample(candidates, limit, seed)
    validate_qa_records(selected)
    stats = {
        "source_rows": int(len(frame)),
        "source_trajectories": int(frame["trail_id"].nunique()),
        "eligible_trajectories": len(candidates),
        "skipped_short": skipped_short,
        "skipped_unmapped": skipped_unmapped,
        "selected_records": len(selected),
        "category_counts": dict(sorted(Counter(r["correct_answer"] for r in selected).items())),
        "difficulty_counts": dict(sorted(Counter(r["difficulty"] for r in selected).items())),
    }
    return selected, stats


def build_dataset(
    checkin_paths: Mapping[str, str | Path],
    taxonomy_path: str | Path,
    output_dir: str | Path,
    *,
    split: str = "test",
    per_city: int = 200,
    seed: int = 5925,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build and export a multi-city Task 1 benchmark."""
    taxonomy_mapping = load_taxonomy_mapping(taxonomy_path)
    all_records: list[dict[str, Any]] = []
    city_stats = {}

    for city_key, checkin_path in checkin_paths.items():
        if city_key not in CITY_SPECS:
            raise ValueError(f"Unknown city key: {city_key}")
        city, _, _ = CITY_SPECS[city_key]
        checkins = pd.read_csv(checkin_path)
        records, stats = build_task1_records(
            checkins,
            city,
            city_key,
            split,
            taxonomy_mapping,
            limit=per_city,
            seed=seed,
        )
        all_records.extend(records)
        city_stats[city] = stats

    validate_qa_records(all_records)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"task1_massive_steps_{split}"
    write_csv_records(all_records, output_dir / f"{stem}.csv")
    write_jsonl(all_records, output_dir / f"{stem}.jsonl")

    summary = {
        "task": TASK_NAME,
        "source_dataset": SOURCE_DATASET,
        "source_split": split,
        "taxonomy": "Foursquare legacy top-level category tree",
        "taxonomy_url": TAXONOMY_URL,
        "taxonomy_sha256": _sha256_file(taxonomy_path),
        "seed": seed,
        "per_city_limit": per_city,
        "total_records": len(all_records),
        "cities": city_stats,
        "source_files": {
            CITY_SPECS[city_key][0]: {
                "filename": Path(checkin_path).name,
                "url": massive_steps_url(city_key, split),
                "sha256": _sha256_file(checkin_path),
            }
            for city_key, checkin_path in checkin_paths.items()
        },
        "category_counts": dict(
            sorted(Counter(r["correct_answer"] for r in all_records).items())
        ),
        "difficulty_counts": dict(
            sorted(Counter(r["difficulty"] for r in all_records).items())
        ),
    }
    with (output_dir / f"{stem}_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=True, indent=2, sort_keys=True)
        file.write("\n")
    return all_records, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build dataset-backed Task 1 QA from Massive-STEPS."
    )
    parser.add_argument("--raw-dir", default="data/raw/massive_steps")
    parser.add_argument("--output-dir", default="data/benchmark/task1_massive_steps")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--per-city", type=int, default=200)
    parser.add_argument("--seed", type=int, default=5925)
    parser.add_argument("--cities", nargs="*", default=list(CITY_SPECS))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    unknown_cities = sorted(set(args.cities).difference(CITY_SPECS))
    if unknown_cities:
        parser.error(f"Unknown city keys: {', '.join(unknown_cities)}")
    if args.per_city < 1:
        parser.error("--per-city must be at least 1")

    raw_dir = Path(args.raw_dir)
    taxonomy_path = raw_dir / "foursquare-taxonomy.json"
    checkin_paths = {
        city_key: raw_dir / f"{CITY_SPECS[city_key][2]}_checkins_{args.split}.csv"
        for city_key in args.cities
    }
    if args.download or args.force_download:
        taxonomy_path, checkin_paths = download_sources(
            raw_dir,
            args.cities,
            args.split,
            force=args.force_download,
        )

    missing_paths = [
        str(path)
        for path in [taxonomy_path, *checkin_paths.values()]
        if not path.exists()
    ]
    if missing_paths:
        parser.error(
            "Missing source files; rerun with --download: " + ", ".join(missing_paths)
        )

    records, summary = build_dataset(
        checkin_paths,
        taxonomy_path,
        args.output_dir,
        split=args.split,
        per_city=args.per_city,
        seed=args.seed,
    )
    print(
        f"Wrote {len(records)} records across {len(summary['cities'])} cities "
        f"to {args.output_dir}."
    )


if __name__ == "__main__":
    main()
