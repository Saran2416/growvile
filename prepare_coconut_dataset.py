"""Merge the raw coconut datasets into one ImageFolder tree.

Two Kaggle datasets are used, and each contributes the classes the other one
lacks or covers less well:

``nagabushan/coconut-tree-disease-dataset``
    Bud Rot, Gray Leaf Spot, Leaf Rot, Stem Bleeding.
``amandapriyashantha/coconut-leaf-diseases``
    CCI_Caterpillars (pest) and Healthy_Leaves.

The second dataset also contains Gray Leaf Spot and Leaf Rot, but those are
copies of the first dataset's images, so they are skipped.  ``Bud Root
Dropping`` is deliberately excluded: it is a dataset label with no documented
pathogen or treatment, and this project must not invent either.

``negatives`` supplies the ``Not_Coconut`` class from ``fetch_negatives.py`` so
that the closed-set classifier can reject leaves that are not coconut.

Images are hard-linked into ``datasets/coconut/merged/<class>/`` when possible
and copied otherwise, so no extra disk space is used on the same filesystem.

Usage::

    .venv/bin/python prepare_coconut_dataset.py

    # verify the layout without writing anything
    .venv/bin/python prepare_coconut_dataset.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Which source dataset supplies which merged class.
CLASS_SOURCES = {
    "nagabushan": ["Bud Rot", "Gray Leaf Spot", "Leaf Rot", "Stem Bleeding"],
    "amandapriyashantha": ["CCI_Caterpillars", "Healthy_Leaves"],
    "negatives": ["Not_Coconut"],
}

EXCLUDED_CLASSES = {
    "Bud Root Dropping": (
        "excluded: the dataset gives no pathogen and no published aetiology, "
        "so cause and cure cannot be stated reliably"
    ),
}


def find_class_dir(source_root: Path, class_name: str) -> Path | None:
    """Find ``<...>/<class_name>/`` below a dataset root."""
    if not source_root.exists():
        return None
    for current, dirs, _files in os.walk(source_root):
        if Path(current).name != class_name:
            continue
        images = [
            item
            for item in Path(current).iterdir()
            if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        ]
        if images:
            return Path(current)
        dirs[:] = []
    return None


def link_or_copy(source: Path, destination: Path, copy: bool) -> None:
    if copy:
        shutil.copy2(source, destination)
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge the raw coconut datasets")
    parser.add_argument("--raw", default="datasets/coconut/raw", help="folder with extracted datasets")
    parser.add_argument("--output", default="datasets/coconut/merged", help="merged ImageFolder root")
    parser.add_argument("--copy", action="store_true", help="copy files instead of hard-linking")
    parser.add_argument("--dry-run", action="store_true", help="report what would be merged")
    args = parser.parse_args()

    raw = Path(args.raw).resolve()
    output = Path(args.output).resolve()
    if not raw.exists():
        raise SystemExit(f"No raw dataset folder at {raw}. Run download_dataset.py first.")

    for excluded, reason in EXCLUDED_CLASSES.items():
        print(f"Skipping {excluded!r}: {reason}")

    total = 0
    summary: dict[str, int] = {}
    missing: list[str] = []

    for source_name, class_names in CLASS_SOURCES.items():
        source_root = raw / source_name
        if not source_root.exists():
            print(f"WARNING: source {source_name!r} not found under {raw}")
            missing.extend(class_names)
            continue
        for class_name in class_names:
            class_dir = find_class_dir(source_root, class_name)
            if class_dir is None:
                print(f"WARNING: no images for {class_name!r} under {source_root}")
                missing.append(class_name)
                continue
            target_dir = output / class_name
            images = sorted(
                item
                for item in class_dir.iterdir()
                if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
            )
            summary[class_name] = len(images)
            total += len(images)
            print(f"{len(images):5d}  {class_name:18s} <- {class_dir}")
            if args.dry_run:
                continue
            target_dir.mkdir(parents=True, exist_ok=True)
            for index, image in enumerate(images):
                destination = target_dir / f"{source_name}_{index:05d}{image.suffix.lower()}"
                if destination.exists():
                    continue
                link_or_copy(image, destination, args.copy)

    print(f"\n{total} images across {len(summary)} classes")
    if not args.dry_run:
        print(f"Merged dataset: {output}")
        print(f"Use it with: .venv/bin/python train_plant.py --data {output}")
    if missing:
        raise SystemExit(f"Missing classes: {', '.join(sorted(set(missing)))}")


if __name__ == "__main__":
    main()
