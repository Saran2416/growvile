"""Measure how well the classifier rejects leaves that are not coconut.

Two things are checked:

1. **Rejection of unseen species.**  Every image under the held-out folder is
   classified; a correct rejection is the ``Not_Coconut`` label.  The holdout
   species were never used in training, so this measures generalisation rather
   than memorisation.
2. **Coconut recall.**  The same 20% validation split that ``train_plant.py``
   uses is reconstructed from the merged dataset, so adding the negative class
   can be checked for collateral damage.

Usage::

    .venv/bin/python evaluate_rejection.py
    .venv/bin/python evaluate_rejection.py --holdout datasets/coconut/negatives_holdout
    .venv/bin/python evaluate_rejection.py --min-rejection 0.9   # exit 1 if below
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def rejection_report(model, holdout: Path) -> tuple[int, int, Counter]:
    """Classify every held-out image and return (rejected, total, mistakes)."""
    rejected = 0
    total = 0
    mistakes: Counter[str] = Counter()
    print("Held-out non-coconut leaves (unseen species)")
    print(f"  {'species':38s} {'images':>6s} {'rejected':>8s} {'rate':>7s}")

    folders = sorted(path for path in holdout.iterdir() if path.is_dir())
    for folder in folders:
        images = list_images(folder)
        if not images:
            continue
        folder_rejected = 0
        folder_total = 0
        for image_path in images:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            result = model.predict(image, topk=1)
            folder_total += 1
            total += 1
            if not result.get("recognized", True):
                rejected += 1
                folder_rejected += 1
            else:
                mistakes[result["label"]] += 1
        rate = folder_rejected / folder_total if folder_total else 0.0
        print(f"  {folder.name:38s} {folder_total:6d} {folder_rejected:8d} {rate:7.1%}")

    overall = rejected / total if total else 0.0
    print(f"  {'OVERALL':38s} {total:6d} {rejected:8d} {overall:7.1%}")
    if mistakes:
        print("  Most common wrong coconut classes:")
        for label, count in mistakes.most_common(5):
            print(f"    {count:4d}  {label}")
    return rejected, total, mistakes


def coconut_recall(model, dataset_root: Path, val_fraction: float, cap: int, seed: int):
    """Rebuild the training validation split and report per-class recall."""
    from torchvision.datasets import ImageFolder

    from train_plant import stratified_split

    dataset = ImageFolder(dataset_root)
    _, val_indices = stratified_split(dataset, val_fraction, cap, seed)
    per_class: dict[str, list[int]] = {}
    for index in val_indices:
        path, label = dataset.samples[index]
        name = dataset.classes[int(label)]
        if name == "Not_Coconut":
            continue
        per_class.setdefault(name, [0, 0])
        per_class[name][1] += 1
        image = cv2.imread(path)
        if image is None:
            continue
        result = model.predict(image, topk=1)
        if result.get("recognized", True) and result["label"] == name:
            per_class[name][0] += 1

    print("\nCoconut recall on the training validation split")
    print(f"  {'class':20s} {'images':>6s} {'correct':>8s} {'recall':>7s}")
    correct_total = 0
    total = 0
    for name in sorted(per_class):
        correct, count = per_class[name]
        correct_total += correct
        total += count
        rate = correct / count if count else 0.0
        print(f"  {name:20s} {count:6d} {correct:8d} {rate:7.1%}")
    overall = correct_total / total if total else 0.0
    print(f"  {'OVERALL':20s} {total:6d} {correct_total:8d} {overall:7.1%}")
    return correct_total, total


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure non-coconut rejection")
    parser.add_argument("--model", default="runs/coconut/best.pt")
    parser.add_argument("--holdout", default="datasets/coconut/negatives_holdout")
    parser.add_argument("--data", default="datasets/coconut/merged")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-per-class", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-rejection",
        type=float,
        default=0.0,
        help="exit with status 1 when the overall rejection rate is below this",
    )
    args = parser.parse_args()

    from plant import PlantClassifier

    model = PlantClassifier(args.model, threads=args.threads)
    rejected, total, _ = rejection_report(model, Path(args.holdout))
    coconut_recall(model, Path(args.data), args.val_fraction, args.max_per_class, args.seed)

    if total == 0:
        raise SystemExit(f"No held-out images found under {args.holdout}")
    rate = rejected / total
    if rate < args.min_rejection:
        raise SystemExit(
            f"Rejection rate {rate:.1%} is below the required {args.min_rejection:.1%}"
        )


if __name__ == "__main__":
    main()
