"""Download non-coconut leaf photos used to train and test rejection.

The coconut classifier is a closed-set model, so without an explicit negative
class it must answer with one of the coconut diseases even when it is shown a
tomato leaf.  This script builds two image sets from public HuggingFace
datasets:

``datasets/coconut/raw/negatives/Not_Coconut/``
    Real-world leaf photos from the PlantDoc dataset (Singh et al., 2020) for
    the species the model is allowed to learn as "not coconut".
``datasets/coconut/negatives_holdout/<species>/``
    PlantDoc species that are *never* used in training, plus the Makerere beans
    test split, so rejection can be measured on unseen plants.

Images are resized to at most 320 px on the long side to keep the download
small; training happens at 224 px.

Examples::

    .venv/bin/python fetch_negatives.py --dry-run
    .venv/bin/python fetch_negatives.py
    .venv/bin/python fetch_negatives.py --force   # re-download everything
"""

from __future__ import annotations

import argparse
import io
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PLANTDOC = "LamTNguyen/PlantDoc"
BEANS = "AI-Lab-Makerere/beans"
ROWS_API = "https://datasets-server.huggingface.co/rows"
MAX_SIDE = 320
JPEG_QUALITY = 90

# Species the negative class may learn.  Everything else is held out.
TRAIN_SPECIES = (
    "Apple_leaf",
    "Apple_Scab_Leaf",
    "Apple_rust_leaf",
    "Bell_pepper_leaf",
    "Bell_pepper_leaf_spot",
    "Corn_Gray_leaf_spot",
    "Corn_leaf_blight",
    "Corn_rust_leaf",
    "grape_leaf",
    "grape_leaf_black_rot",
    "Potato_leaf_early_blight",
    "Potato_leaf_late_blight",
    "Tomato_leaf",
    "Tomato_Early_blight_leaf",
    "Tomato_Septoria_leaf_spot",
    "Tomato_leaf_bacterial_spot",
    "Tomato_leaf_late_blight",
    "Tomato_leaf_mosaic_virus",
    "Tomato_leaf_yellow_virus",
    "Tomato_mold_leaf",
)
HOLDOUT_SPECIES = (
    "Blueberry_leaf",
    "Cherry_leaf",
    "Peach_leaf",
    "Raspberry_leaf",
    "Soyabean_leaf",
    "Squash_Powdery_mildew_leaf",
    "Strawberry_leaf",
)

TRAIN_DIR = Path("datasets/coconut/raw/negatives/Not_Coconut")
HOLDOUT_DIR = Path("datasets/coconut/negatives_holdout")


def fetch_json(url: str, attempts: int = 8, timeout: int = 60) -> dict:
    """GET JSON, backing off on the datasets-server rate limit (HTTP 429)."""
    import requests

    delay = 1.0
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 429:
                raise requests.HTTPError("429 Too Many Requests", response=response)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # network errors, 429, 5xx
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(delay)
            delay = min(delay * 2, 20)
    raise RuntimeError(f"Could not fetch {url}: {last_error}")


def fetch_bytes(url: str, attempts: int = 5, timeout: int = 60) -> bytes:
    import requests

    delay = 1.0
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.content
        except Exception as exc:
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(delay)
            delay = min(delay * 2, 15)
    raise RuntimeError(f"Could not download image: {last_error}")


def dataset_rows(dataset: str, split: str, page_size: int = 100, pause: float = 0.3):
    """Yield ``(row_index, image_url, label_name)`` for a dataset split."""
    info = fetch_json(f"{ROWS_API}?dataset={dataset.replace('/', '%2F')}"
                      f"&config=default&split={split}&offset=0&length=1")
    names = None
    for feature in info.get("features", []):
        if feature.get("name") in ("label", "labels"):
            names = feature["type"]["names"]
    if names is None:
        raise RuntimeError(f"{dataset}/{split} has no label feature")
    total = int(info["num_rows_total"])
    print(f"{dataset}/{split}: {total} rows, {len(names)} labels")

    offset = 0
    while offset < total:
        payload = fetch_json(
            f"{ROWS_API}?dataset={dataset.replace('/', '%2F')}&config=default"
            f"&split={split}&offset={offset}&length={page_size}"
        )
        rows = payload.get("rows", [])
        if not rows:
            break
        for row in rows:
            label = row["row"].get("label", row["row"].get("labels"))
            yield int(row["row_idx"]), row["row"]["image"]["src"], names[int(label)]
        offset += len(rows)
        time.sleep(pause)


def collect(urls_by_label: dict[str, list[tuple[int, str]]], limit: int, seed: int):
    """Pick a reproducible random subset per label."""
    rng = random.Random(seed)
    selected: dict[str, list[tuple[int, str]]] = {}
    for label, items in urls_by_label.items():
        items = items[:]
        rng.shuffle(items)
        selected[label] = items[:limit]
    return selected


def save_image(raw: bytes, destination: Path) -> bool:
    from PIL import Image, UnidentifiedImageError

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return False
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return True


def download_set(
    selected: dict[str, list[tuple[int, str]]],
    output: Path,
    prefix: str,
    force: bool,
    workers: int,
    per_label_dir: bool = False,
) -> int:
    tasks: list[tuple[str, int, str, Path]] = []
    for label, items in selected.items():
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._") or "image"
        folder = output / safe if per_label_dir else output
        for row_index, url in items:
            tasks.append((label, row_index, url, folder / f"{prefix}_{safe}_{row_index:05d}.jpg"))

    pending = [task for task in tasks if force or not task[3].exists()]
    skipped = len(tasks) - len(pending)
    print(f"{output}: {len(tasks)} images ({skipped} already present)")

    saved = 0
    failed = 0

    def work(task):
        nonlocal saved, failed
        _label, _row, url, destination = task
        try:
            raw = fetch_bytes(url)
        except RuntimeError as exc:
            print(f"  skip {destination.name}: {exc}")
            failed += 1
            return
        if save_image(raw, destination):
            saved += 1
        else:
            failed += 1

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for index, _ in enumerate(pool.map(work, pending), start=1):
            if index % 50 == 0:
                print(f"  {index}/{len(pending)} downloaded")

    print(f"  saved {saved}, failed {failed}")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Download non-coconut leaf negatives")
    parser.add_argument("--plantdoc-train-per-label", type=int, default=0,
                        help="images per training label (0 = all)")
    parser.add_argument("--plantdoc-holdout-per-label", type=int, default=100,
                        help="images per held-out label (0 = all)")
    parser.add_argument("--beans-per-label", type=int, default=50,
                        help="held-out bean images per class (0 = all)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="download even if the file exists")
    parser.add_argument("--dry-run", action="store_true", help="only report counts")
    args = parser.parse_args()

    train_urls: dict[str, list[tuple[int, str]]] = {}
    holdout_urls: dict[str, list[tuple[int, str]]] = {}
    for index, url, label in dataset_rows(PLANTDOC, "train"):
        if label in TRAIN_SPECIES:
            train_urls.setdefault(label, []).append((index, url))
        elif label in HOLDOUT_SPECIES:
            holdout_urls.setdefault(label, []).append((index, url))

    beans_urls: dict[str, list[tuple[int, str]]] = {}
    for index, url, label in dataset_rows(BEANS, "test"):
        beans_urls.setdefault(label, []).append((index, url))

    train_limit = args.plantdoc_train_per_label or 10**9
    holdout_limit = args.plantdoc_holdout_per_label or 10**9
    beans_limit = args.beans_per_label or 10**9
    train_selected = collect(train_urls, train_limit, args.seed)
    holdout_selected = collect(holdout_urls, holdout_limit, args.seed)
    beans_selected = collect(beans_urls, beans_limit, args.seed)

    print("\nTraining negatives (PlantDoc, seen species):")
    for label, items in sorted(train_selected.items()):
        print(f"  {len(items):3d}  {label}")
    print("\nHeld-out negatives (PlantDoc, unseen species):")
    for label, items in sorted(holdout_selected.items()):
        print(f"  {len(items):3d}  {label}")
    print("\nHeld-out negatives (Makerere beans):")
    for label, items in sorted(beans_selected.items()):
        print(f"  {len(items):3d}  {label}")

    total_train = sum(len(items) for items in train_selected.values())
    total_holdout = sum(len(items) for items in holdout_selected.values()) + sum(
        len(items) for items in beans_selected.values()
    )
    print(f"\nTotal: {total_train} training, {total_holdout} held out")
    if args.dry_run:
        return

    download_set(train_selected, TRAIN_DIR, "plantdoc", args.force, args.workers)
    download_set(
        holdout_selected, HOLDOUT_DIR, "plantdoc", args.force, args.workers, per_label_dir=True
    )
    download_set(beans_selected, HOLDOUT_DIR / "beans", "beans", args.force, args.workers)

    sources = {
        "plantdoc": "https://huggingface.co/datasets/LamTNguyen/PlantDoc",
        "beans": "https://huggingface.co/datasets/AI-Lab-Makerere/beans",
    }
    (TRAIN_DIR.parent / "manifest.json").write_text(
        json.dumps(
            {
                "sources": sources,
                "species": sorted(train_selected),
                "images": total_train,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (HOLDOUT_DIR / "manifest.json").write_text(
        json.dumps(
            {
                "sources": sources,
                "plantdoc_species": sorted(holdout_selected),
                "beans_species": sorted(beans_selected),
                "plantdoc_images": sum(len(items) for items in holdout_selected.values()),
                "beans_images": sum(len(items) for items in beans_selected.values()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nTraining negatives: {TRAIN_DIR}")
    print(f"Held-out negatives: {HOLDOUT_DIR}")
    print("Next: .venv/bin/python prepare_coconut_dataset.py")


if __name__ == "__main__":
    main()
