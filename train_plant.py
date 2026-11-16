"""Train the plant species + leaf disease classifier on CPU.

The model is a small ImageNet-pretrained backbone with a fresh classification
head.  On a CPU-only machine the practical recipe is:

1. train only the head with the backbone frozen (fast), then
2. unfreeze the whole network for a couple of low-learning-rate epochs.

Example::

    .venv/bin/python train_plant.py \
        --data datasets/coconut/merged \
        --epochs 6 --freeze-epochs 4 --max-per-class 300

The best checkpoint is written to ``runs/coconut/best.pt`` and is used by
``plant_camera.py`` and ``inference.py --plant``.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from plant import (
    ARCHITECTURES,
    build_eval_transform,
    build_model,
    build_train_transform,
    classifier_parameters,
    freeze_backbone,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_imagefolder_root(root: Path) -> Path:
    """Find the folder whose immediate children are the class folders."""
    root = root.expanduser().resolve()

    def score(candidate: Path) -> int:
        try:
            children = [child for child in candidate.iterdir() if child.is_dir()]
        except OSError:
            return 0
        return sum(
            1
            for child in children
            if any(file.suffix.lower() in IMAGE_SUFFIXES for file in child.iterdir() if file.is_file())
        )

    best_root, best_score = root, score(root)
    if best_score >= 2:
        return best_root

    queue: list[tuple[Path, int]] = [(root, 0)]
    seen: set[Path] = set()
    while queue:
        current, depth = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        current_score = score(current)
        if current_score > best_score:
            best_root, best_score = current, current_score
        if depth >= 3:
            continue
        try:
            children = sorted(child for child in current.iterdir() if child.is_dir())
        except OSError:
            continue
        queue.extend((child, depth + 1) for child in children)

    if best_score < 2:
        raise SystemExit(
            f"No ImageFolder layout found under {root}. Expected "
            "<root>/<class-name>/<images>."
        )
    return best_root


class TransformSubset:
    """Apply a transform to a subset of an ImageFolder dataset."""

    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform
        self.classes = dataset.classes

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        image, label = self.dataset[self.indices[position]]
        return self.transform(image), label


def stratified_split(dataset, val_fraction: float, max_per_class: int, seed: int):
    """Split indices per class so every class appears in train and validation."""
    rng = random.Random(seed)
    per_class: dict[int, list[int]] = {}
    for index, (_, label) in enumerate(dataset.samples):
        per_class.setdefault(int(label), []).append(index)

    train_indices: list[int] = []
    val_indices: list[int] = []
    for label in sorted(per_class):
        indices = per_class[label][:]
        rng.shuffle(indices)
        if max_per_class and len(indices) > max_per_class:
            indices = indices[:max_per_class]
        if len(indices) < 2:
            train_indices.extend(indices)
            continue
        val_count = max(1, round(len(indices) * val_fraction))
        val_count = min(val_count, len(indices) - 1)
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def evaluate(model, loader, device: str) -> tuple[float, float, float]:
    """Return (loss, top1 accuracy, top3 accuracy)."""
    import torch
    from torch import nn

    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    correct1 = 0
    correct3 = 0
    total = 0
    with torch.inference_mode():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            total_loss += float(criterion(logits, targets)) * targets.size(0)
            top3 = logits.topk(min(3, logits.shape[1]), dim=1).indices
            correct1 += int((top3[:, 0] == targets).sum())
            correct3 += int((top3 == targets.unsqueeze(1)).any(dim=1).sum())
            total += targets.size(0)
    if total == 0:
        return float("nan"), 0.0, 0.0
    return total_loss / total, correct1 / total, correct3 / total


def train_epoch(model, loader, optimizer, criterion, device: str) -> float:
    from tqdm import tqdm

    model.train()
    running = 0.0
    total = 0
    progress = tqdm(loader, desc="  train", unit="batch", leave=False)
    for images, targets in progress:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        running += float(loss) * targets.size(0)
        total += targets.size(0)
        progress.set_postfix(loss=f"{running / max(1, total):.3f}")
    return running / max(1, total)


def class_weights(dataset, train_indices, mode: str):
    """Build CrossEntropyLoss weights from the training split.

    ``sqrt`` softens the correction, which keeps the majority classes usable
    while still giving a small class such as Healthy_Leaves (123 images) a
    voice.  ``inverse`` is the classic full inverse-frequency weighting.
    """
    import torch

    counts = [0] * len(dataset.classes)
    for index in train_indices:
        counts[int(dataset.samples[index][1])] += 1

    present = [count for count in counts if count > 0]
    if mode == "none" or not present:
        return None

    if mode == "inverse":
        weights = [len(train_indices) / (len(present) * count) if count else 0.0 for count in counts]
    else:
        smallest = min(present)
        weights = [(smallest / count) ** 0.5 if count else 0.0 for count in counts]
    return torch.tensor(weights, dtype=torch.float32)


def per_class_metrics(model, loader, class_names, device: str) -> dict[str, dict[str, float]]:
    """Confusion-matrix based precision / recall / F1 per class."""
    import torch

    size = len(class_names)
    confusion = [[0] * size for _ in range(size)]
    model.eval()
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images.to(device))
            predictions = logits.argmax(dim=1).cpu()
            for target, prediction in zip(targets.tolist(), predictions.tolist()):
                confusion[target][prediction] += 1

    metrics: dict[str, dict[str, float]] = {}
    for index, name in enumerate(class_names):
        true_positive = confusion[index][index]
        predicted = sum(row[index] for row in confusion)
        actual = sum(confusion[index])
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": actual,
        }
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a plant disease classifier on CPU")
    parser.add_argument("--data", required=True, help="dataset root with <class>/<image> folders")
    parser.add_argument("--arch", default="mobilenet_v3_small", choices=ARCHITECTURES)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=4,
        help="epochs that train only the head (fast); the rest fine-tune everything",
    )
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=300,
        help="cap images per class for a quick CPU run (0 = use everything)",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument(
        "--class-weights",
        choices=("none", "sqrt", "inverse"),
        default="sqrt",
        help="compensate for imbalanced classes (Healthy_Leaves has few images)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="runs/coconut")
    parser.add_argument(
        "--init",
        default=None,
        help="warm-start from an existing checkpoint instead of ImageNet weights",
    )
    parser.add_argument(
        "--augment",
        choices=("basic", "strong"),
        default="basic",
        help="strong adds vertical flips, blur and random erasing for continuation runs",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--inspect", action="store_true", help="only print class counts and exit")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    root = find_imagefolder_root(Path(args.data))
    dataset = ImageFolder(root)
    train_indices, val_indices = stratified_split(
        dataset, args.val_fraction, args.max_per_class, args.seed
    )

    print(f"Dataset : {root}")
    print(f"Classes : {len(dataset.classes)}")
    print(f"Train   : {len(train_indices)} images")
    print(f"Val     : {len(val_indices)} images")
    print(f"Device  : {device} | threads={args.threads} | arch={args.arch}")

    if args.inspect:
        counts = {dataset.classes[label]: 0 for label in range(len(dataset.classes))}
        for _, label in dataset.samples:
            counts[dataset.classes[label]] += 1
        for name, count in counts.items():
            print(f"  {count:6d}  {name}")
        return

    train_set = TransformSubset(
        dataset,
        train_indices,
        build_train_transform(args.imgsz, strong=args.augment == "strong"),
    )
    val_set = TransformSubset(dataset, val_indices, build_eval_transform(args.imgsz))
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.startswith("cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.startswith("cuda"),
    )

    model = build_model(args.arch, len(dataset.classes), pretrained=True).to(device)
    if args.init:
        init_path = Path(args.init)
        if not init_path.exists():
            raise SystemExit(f"No checkpoint at {init_path}")
        checkpoint = torch.load(init_path, map_location=device, weights_only=True)
        if checkpoint.get("arch") not in (None, args.arch):
            raise SystemExit(
                f"{init_path} was trained with {checkpoint['arch']}, not {args.arch}"
            )
        if checkpoint.get("classes") not in (None, dataset.classes):
            raise SystemExit(
                f"{init_path} classes do not match the dataset classes"
            )
        model.load_state_dict(checkpoint["state_dict"])
        print(f"Init    : {init_path} (top1 {checkpoint.get('val_accuracy', 0) * 100:.2f}%)")
    weights = class_weights(dataset, train_indices, args.class_weights)
    if weights is not None:
        weights = weights.to(device)
        print("Weights : " + ", ".join(f"{name} {value:.2f}" for name, value in zip(dataset.classes, weights.tolist())))
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    optimizer = torch.optim.AdamW(classifier_parameters(model), lr=args.lr_head, weight_decay=1e-4)
    unfrozen = False
    best_accuracy = 0.0
    best_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, float]] = []
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    for epoch in range(1, args.epochs + 1):
        if not unfrozen and epoch > args.freeze_epochs:
            freeze_backbone(model, frozen=False)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr_finetune, weight_decay=1e-4
            )
            unfrozen = True
            print(f"Epoch {epoch}: backbone unfrozen, fine-tuning at lr={args.lr_finetune}")

        freeze_backbone(model, frozen=not unfrozen)
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, top1, top3 = evaluate(model, val_loader, device)
        elapsed = time.monotonic() - started
        print(
            f"Epoch {epoch}/{args.epochs} | train {train_loss:.3f} | "
            f"val {val_loss:.3f} | top1 {top1 * 100:.2f}% | top3 {top3 * 100:.2f}% | "
            f"{elapsed:.0f}s"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "top1": round(top1, 4),
                "top3": round(top3, 4),
            }
        )

        # Prefer higher top-1 accuracy, then lower validation loss. Validation
        # accuracy saturates quickly on this dataset, so the loss tie-break
        # keeps training while the checkpoint is still improving.
        if top1 > best_accuracy or (top1 == best_accuracy and val_loss < best_loss):
            best_accuracy = top1
            best_loss = val_loss
            best_epoch = epoch
            payload = {
                "arch": args.arch,
                "classes": dataset.classes,
                "imgsz": args.imgsz,
                "state_dict": model.state_dict(),
                "val_accuracy": round(top1, 4),
                "epoch": epoch,
            }
            torch.save(payload, output / "best.pt")
            print(f"  saved {output / 'best.pt'} (top1 {top1 * 100:.2f}%)")
        elif epoch - best_epoch >= args.patience:
            print(f"Early stopping: no improvement for {args.patience} epochs.")
            break

    best_path = output / "best.pt"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
    class_report = per_class_metrics(model, val_loader, dataset.classes, device)
    print("\nPer-class validation metrics:")
    print(f"  {'class':20s} {'precision':>9s} {'recall':>7s} {'f1':>6s} {'support':>8s}")
    for name, values in class_report.items():
        print(
            f"  {name:20s} {values['precision']:9.3f} {values['recall']:7.3f} "
            f"{values['f1']:6.3f} {values['support']:8d}"
        )

    torch.save(
        {
            "arch": args.arch,
            "classes": dataset.classes,
            "imgsz": args.imgsz,
            "state_dict": model.state_dict(),
            "val_accuracy": round(best_accuracy, 4),
        },
        output / "last.pt",
    )
    (output / "metrics.json").write_text(
        json.dumps(
            {
                "arch": args.arch,
                "imgsz": args.imgsz,
                "best_epoch": best_epoch,
                "best_top1": round(best_accuracy, 4),
                "train_images": len(train_indices),
                "val_images": len(val_indices),
                "classes": dataset.classes,
                "class_weights": args.class_weights,
                "per_class": class_report,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Best top1: {best_accuracy * 100:.2f}% (epoch {best_epoch})")
    print(f"Checkpoint: {output / 'best.pt'}")
    print("Try it    : .venv/bin/python plant_camera.py --image leaf.jpg")


if __name__ == "__main__":
    main()
