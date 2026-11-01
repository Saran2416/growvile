"""Fine-tune a small YOLO model on a custom (optionally Kaggle) dataset.

Memory-friendly defaults are used on purpose: the nano checkpoint, a small
batch, two data-loader workers, and no dataset caching.

Typical flow::

    .venv/bin/python -m pip install kaggle
    .venv/bin/python train.py --kaggle owner/dataset-slug --epochs 50

    # or, if the dataset is already on disk:
    .venv/bin/python train.py --data datasets/my-objects/data.yaml --epochs 50

The best checkpoint is written to
``runs/detect/train/weights/best.pt`` by default, which is exactly the path
used by the README examples.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from download_dataset import (
    download_kaggle_dataset,
    find_data_yaml,
    prepare_data_yaml,
)


def default_device() -> str:
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on a custom dataset")
    parser.add_argument("--data", help="path to data.yaml (or a dataset directory)")
    parser.add_argument("--kaggle", help="Kaggle slug to download first, e.g. owner/dataset")
    parser.add_argument("--dataset-dir", default="datasets", help="where Kaggle data is stored")
    parser.add_argument("--model", default=os.getenv("YOLO_MODEL", "yolov8n.pt"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8, help="small batch keeps RAM low")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--freeze",
        type=int,
        default=0,
        help="freeze the first N backbone layers; speeds up CPU training",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="fraction of the training set to use per epoch (0-1)",
    )
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="train")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache", action="store_true", help="cache images (uses much more RAM)")
    return parser


def resolve_data_yaml(args: argparse.Namespace) -> Path:
    if args.kaggle:
        root = download_kaggle_dataset(args.kaggle, args.dataset_dir)
        found = find_data_yaml(root)
        if found is None:
            raise SystemExit("No data.yaml found in the downloaded dataset.")
        return prepare_data_yaml(found)
    if not args.data:
        raise SystemExit("Provide --data path/to/data.yaml or --kaggle owner/dataset")
    data = Path(args.data).expanduser()
    if data.is_dir():
        found = find_data_yaml(data)
        if found is None:
            raise SystemExit(f"No data.yaml found under {data}")
        return prepare_data_yaml(found)
    if not data.exists():
        raise SystemExit(f"Dataset config not found: {data}")
    return prepare_data_yaml(data)


def main() -> None:
    args = build_parser().parse_args()
    data_yaml = resolve_data_yaml(args)

    try:
        import torch

        if args.threads > 0:
            torch.set_num_threads(args.threads)
    except Exception:
        pass

    from ultralytics import YOLO

    model = YOLO(args.model)
    print(f"Training {args.model} on {data_yaml} (device={args.device})", file=sys.stderr)
    if args.device == "cpu":
        print(
            "No GPU selected: training on CPU. Expect hours rather than minutes. "
            "Use --imgsz 416, --batch 4-8, --freeze 10 and a smaller dataset.",
            file=sys.stderr,
        )
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        device=args.device,
        freeze=args.freeze,
        fraction=args.fraction,
        project=args.project,
        name=args.name,
        exist_ok=True,
        cache=args.cache,
        amp=args.device not in {"cpu", "mps"},
        close_mosaic=10,
        resume=args.resume,
    )

    save_dir = Path(getattr(results, "save_dir", Path(args.project) / args.name))
    best = save_dir / "weights" / "best.pt"
    print(f"Best weights: {best}")
    print(f"Use with: .venv/bin/python inference.py samples/bus.jpg --model {best}")


if __name__ == "__main__":
    main()
