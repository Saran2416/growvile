"""Generate featureless images for the ``Not_Coconut`` training class.

A webcam can face a wall, a desk, the sky or a lens cap.  The coconut
classifier is a closed-set model, so without examples of such frames it can
answer ``Healthy`` with low confidence instead of refusing.  This script writes
deterministic synthetic negatives (flat colours, gradients, noise, stripes and
smooth colour fields) next to the downloaded PlantDoc negatives.

They contain no leaf texture on purpose: the model should learn "no foliage
here", not "textured thing is not coconut".

Usage::

    .venv/bin/python make_synthetic_negatives.py
    .venv/bin/python make_synthetic_negatives.py --count 300 --dry-run
    .venv/bin/python make_synthetic_negatives.py --clean   # remove them again
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

DEFAULT_OUTPUT = Path("datasets/coconut/raw/negatives/Not_Coconut")
SIZE = 320
PREFIX = "synthetic_"


def _flat(rng: np.random.Generator) -> np.ndarray:
    color = rng.integers(0, 256, size=3)
    image = np.empty((SIZE, SIZE, 3), dtype=np.float32)
    image[:] = color
    image += rng.normal(0, 3, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def _gradient(rng: np.random.Generator) -> np.ndarray:
    start = rng.integers(0, 256, size=3).astype(np.float32)
    end = rng.integers(0, 256, size=3).astype(np.float32)
    axis = int(rng.integers(0, 3))
    ramp = np.linspace(0.0, 1.0, SIZE, dtype=np.float32)
    if axis == 1:
        ramp = ramp.reshape(1, SIZE, 1)
    elif axis == 2:
        ramp = (ramp.reshape(SIZE, 1) + ramp.reshape(1, SIZE)) / 2
        ramp = ramp[:, :, None]
    else:
        ramp = ramp.reshape(SIZE, 1, 1)
    image = start + (end - start) * ramp
    image = np.broadcast_to(image, (SIZE, SIZE, 3)).astype(np.float32)
    image += rng.normal(0, 4, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def _noise(rng: np.random.Generator) -> np.ndarray:
    image = rng.integers(0, 256, size=(SIZE, SIZE, 3), dtype=np.uint8)
    if rng.random() < 0.5:
        image = cv2.GaussianBlur(image, (5, 5), 0)
    return image


def _smooth_field(rng: np.random.Generator) -> np.ndarray:
    low = rng.integers(0, 256, size=(rng.integers(3, 7), rng.integers(3, 7), 3)).astype(np.uint8)
    image = cv2.resize(low, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    image += rng.normal(0, 5, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def _pattern(rng: np.random.Generator) -> np.ndarray:
    period = int(rng.integers(12, 48))
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    if rng.random() < 0.5:
        mask = ((yy // period) + (xx // period)) % 2
    else:
        mask = (xx // period) % 2
    first = rng.integers(0, 256, size=3)
    second = rng.integers(0, 256, size=3)
    image = np.where(mask[:, :, None].astype(bool), first, second).astype(np.float32)
    image += rng.normal(0, 4, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def _solid(rng: np.random.Generator) -> np.ndarray:
    value = int(rng.choice([0, 32, 64, 128, 192, 224, 255]))
    image = np.full((SIZE, SIZE, 3), value, dtype=np.uint8)
    return image


GENERATORS = (_flat, _gradient, _noise, _smooth_field, _pattern, _solid)


def generate(output: Path, count: int, seed: int) -> int:
    rng = np.random.default_rng(seed)
    output.mkdir(parents=True, exist_ok=True)
    written = 0
    for index in range(count):
        image = GENERATORS[index % len(GENERATORS)](rng)
        path = output / f"{PREFIX}{index:05d}.jpg"
        if cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 90]):
            written += 1
    return written


def clean(output: Path) -> int:
    removed = 0
    for path in sorted(output.glob(f"{PREFIX}*.jpg")):
        path.unlink()
        removed += 1
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate featureless Not_Coconut negatives")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--count", type=int, default=460)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clean", action="store_true", help="remove generated images")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if args.clean:
        print(f"removed {clean(output)} synthetic images from {output}")
        return
    if args.dry_run:
        print(f"would write {args.count} synthetic images to {output}")
        return
    written = generate(output, args.count, args.seed)
    print(f"wrote {written} synthetic images to {output}")
    print("Next: .venv/bin/python prepare_coconut_dataset.py")


if __name__ == "__main__":
    main()
